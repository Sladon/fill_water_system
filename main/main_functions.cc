/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "esp_log.h"
#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "bottle_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"

#include "driver/gpio.h"

#define RELAY_PIN GPIO_NUM_14
#define TRIGGER_PIN GPIO_NUM_3
#define ECHO_PIN GPIO_NUM_46
#define TRIGGER_HIGH_TIME_US 10

#define SPEED_OF_SOUND_CM_PER_MICROSECOND 0.0343
#define MICROSECONDS_TO_CM_DIVISOR 2
#define PULSE_DURATION_MICROSECONDS 10
#define TRIG_PIN_DELAY_MICROSECONDS 2

#define DISTANCE_MIN_RANGE 10
#define DISTANCE_MAX_RANGE 16

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
float distance = -1;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 40 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 81 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_bottle_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

#ifndef CLI_ONLY_INFERENCE
  // Initialize Camera
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}

#ifndef CLI_ONLY_INFERENCE
void loop() {
  if (distance >= DISTANCE_MIN_RANGE && distance <= DISTANCE_MAX_RANGE) {
    // Get image from provider.
    if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8)) {
      MicroPrintf("Image capture failed.");
    }

    // Run the model on this input and make sure it succeeds.
    if (kTfLiteOk != interpreter->Invoke()) {
      MicroPrintf("Invoke failed.");
    }

    TfLiteTensor* output = interpreter->output(0);

    // Process the inference results.
    int8_t bottle_score = output->data.uint8[kBottleIndex];

    float bottle_score_f =
        (bottle_score - output->params.zero_point) * output->params.scale;
    RespondToDetection(bottle_score_f, 1-bottle_score_f);
    if (bottle_score_f > 0.6) {
      gpio_set_level(RELAY_PIN, 1); // Turn on relay
    } else {
      gpio_set_level(RELAY_PIN, 0); // Turn off relay
    }
  } else {
    gpio_set_level(RELAY_PIN, 0);
  }
  
  vTaskDelay(pdMS_TO_TICKS(10)); // to avoid watchdog trigger
}
#endif


#if defined(COLLECT_CPU_STATS)
  long long total_time = 0;
  long long start_time = 0;
  extern long long softmax_total_time;
  extern long long dc_total_time;
  extern long long conv_total_time;
  extern long long fc_total_time;
  extern long long pooling_total_time;
  extern long long add_total_time;
  extern long long mul_total_time;
#endif

void run_inference(void *ptr) {
  /* Convert from uint8 picture data to int8 */
  for (int i = 0; i < kNumCols * kNumRows; i++) {
    input->data.int8[i] = ((uint8_t *) ptr)[i] ^ 0x80;
  }

#if defined(COLLECT_CPU_STATS)
  long long start_time = esp_timer_get_time();
#endif
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

#if defined(COLLECT_CPU_STATS)
  long long total_time = (esp_timer_get_time() - start_time);
  printf("Total time = %lld\n", total_time / 1000);
  //printf("Softmax time = %lld\n", softmax_total_time / 1000);
  printf("FC time = %lld\n", fc_total_time / 1000);
  printf("DC time = %lld\n", dc_total_time / 1000);
  printf("conv time = %lld\n", conv_total_time / 1000);
  printf("Pooling time = %lld\n", pooling_total_time / 1000);
  printf("add time = %lld\n", add_total_time / 1000);
  printf("mul time = %lld\n", mul_total_time / 1000);

  /* Reset times */
  total_time = 0;
  //softmax_total_time = 0;
  dc_total_time = 0;
  conv_total_time = 0;
  fc_total_time = 0;
  pooling_total_time = 0;
  add_total_time = 0;
  mul_total_time = 0;
#endif

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t bottle_score = output->data.uint8[kBottleIndex];
  int8_t no_bottle_score = output->data.uint8[kNotABottleIndex];

  float bottle_score_f =
      (bottle_score - output->params.zero_point) * output->params.scale;
  float no_bottle_score_f =
      (no_bottle_score - output->params.zero_point) * output->params.scale;
  RespondToDetection(bottle_score_f, no_bottle_score_f);
}

portMUX_TYPE distanceLock;
void hardware_init() {
  gpio_set_direction(RELAY_PIN, GPIO_MODE_OUTPUT);
  gpio_set_direction(TRIGGER_PIN, GPIO_MODE_OUTPUT);
  gpio_set_direction(ECHO_PIN, GPIO_MODE_INPUT);
  spinlock_initialize(&distanceLock);
}

float hcsr04_measure() {    
  // Send a 10us pulse to trigger
  portENTER_CRITICAL(&distanceLock);
  gpio_set_level(TRIGGER_PIN, 0);
  esp_rom_delay_us(2);
    gpio_set_level(TRIGGER_PIN, 1);
    esp_rom_delay_us(TRIGGER_HIGH_TIME_US);
    gpio_set_level(TRIGGER_PIN, 0);
  
  // Wait for echo to be high
  int timeout = 3000;  // Maximum waiting time in microseconds
  while (gpio_get_level(ECHO_PIN) == 0 && timeout > 0) {
    esp_rom_delay_us(1);  // Small delay to prevent overloading CPU
    // taskYIELD();      // Yield to reset WDT and allow other tasks to run
    --timeout;
  }
  int64_t start_time = esp_timer_get_time();
  portEXIT_CRITICAL(&distanceLock);

  if (timeout == 0) {
    ESP_LOGW("DistanceTask", "Timeout waiting for ECHO_PIN to go high");
    return -1;
  }

  // Wait for ECHO_PIN to go low (end of the echo pulse)
  portENTER_CRITICAL(&distanceLock);
  timeout = 3000;
  while (gpio_get_level(ECHO_PIN) == 1 && timeout > 0) {
    esp_rom_delay_us(1);  // Small delay to prevent overloading CPU
    // taskYIELD();      // Yield to reset WDT and allow other tasks to run
    --timeout;
  }
  int64_t end_time = esp_timer_get_time();
  portEXIT_CRITICAL(&distanceLock);

  if (timeout == 0) {
    ESP_LOGW("DistanceTask", "Timeout waiting for ECHO_PIN to go low");
    return -1;
  }

  long duration = end_time - start_time;
  distance = (duration / MICROSECONDS_TO_CM_DIVISOR) * SPEED_OF_SOUND_CM_PER_MICROSECOND;
  
  return distance;
}

void hcsr04_task(void *pvParameters) {
  while(1) {
    distance = hcsr04_measure();
    ESP_LOGI("HCSR04", "Distance: %.2f cm", distance);
    vTaskDelay(pdMS_TO_TICKS(1000)); // Delay 1 second
  }
}