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

#ifndef CLI_ONLY_INFERENCE
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
#endif

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
#ifndef CLI_ONLY_INFERENCE
float distance = -1;
#endif

constexpr int kTensorArenaSize = 95 * 1024;
static uint8_t *tensor_arena;
}

#ifndef CLI_ONLY_INFERENCE
portMUX_TYPE distanceLock;
#endif

void setup() {
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

  static tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddLogistic();

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);

#ifndef CLI_ONLY_INFERENCE
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }

  gpio_set_direction(RELAY_PIN, GPIO_MODE_OUTPUT);
  gpio_set_direction(TRIGGER_PIN, GPIO_MODE_OUTPUT);
  gpio_set_direction(ECHO_PIN, GPIO_MODE_INPUT);
  spinlock_initialize(&distanceLock);
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
    for (int i = 0; i < kNumCols * kNumRows; i++) {
      printf("%d, ", input->data.int8[i]);
    }
    if (kTfLiteOk != interpreter->Invoke()) {
      MicroPrintf("Invoke failed.");
    }

    TfLiteTensor* output = interpreter->output(0);

    // Process the inference results.
    int8_t bottle_score = output->data.uint8[kBottleIndex];

    float bottle_score_f =
        1-(bottle_score - output->params.zero_point) * output->params.scale;
    RespondToDetection(bottle_score_f);
    if (bottle_score_f > 0.6) {
      gpio_set_level(RELAY_PIN, 1); 
    } else {
      gpio_set_level(RELAY_PIN, 0);
    }
  } else {
    gpio_set_level(RELAY_PIN, 0);
  }
  
  vTaskDelay(pdMS_TO_TICKS(10));
}
#endif


#if defined(COLLECT_CPU_STATS)
  long long total_time = 0;
  long long start_time = 0;
  extern long long conv_total_time;
  extern long long fc_total_time;
  extern long long pooling_total_time;
#endif

void run_inference(void *ptr) {
  /* Convert from uint8 picture data to int8 */
  for (int i = 0; i < kNumCols * kNumRows; i++) {
    input->data.int8[i] = ((uint8_t *) ptr)[i];
    printf("%d, ", input->data.int8[i]);
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
  printf("FC time = %lld\n", fc_total_time / 1000);
  printf("conv time = %lld\n", conv_total_time / 1000);
  printf("Pooling time = %lld\n", pooling_total_time / 1000);
  /* Reset times */
  total_time = 0;
  conv_total_time = 0;
  fc_total_time = 0;
  pooling_total_time = 0;
#endif

  TfLiteTensor* output = interpreter->output(0);

  int8_t bottle_score = output->data.uint8[kBottleIndex];

  float bottle_score_f =
      1-(bottle_score - output->params.zero_point) * output->params.scale;
  RespondToDetection(bottle_score_f);
}

#ifndef CLI_ONLY_INFERENCE
float hcsr04_measure() {    
  portENTER_CRITICAL(&distanceLock);
  gpio_set_level(TRIGGER_PIN, 0);
  esp_rom_delay_us(2);
    gpio_set_level(TRIGGER_PIN, 1);
    esp_rom_delay_us(TRIGGER_HIGH_TIME_US);
    gpio_set_level(TRIGGER_PIN, 0);

  int timeout = 3000;
  while (gpio_get_level(ECHO_PIN) == 0 && timeout > 0) {
    esp_rom_delay_us(1);
    --timeout;
  }
  int64_t start_time = esp_timer_get_time();
  portEXIT_CRITICAL(&distanceLock);

  if (timeout == 0) {
    ESP_LOGW("DistanceTask", "Timeout waiting for ECHO_PIN to go high");
    return -1;
  }

  portENTER_CRITICAL(&distanceLock);
  timeout = 3000;
  while (gpio_get_level(ECHO_PIN) == 1 && timeout > 0) {
    esp_rom_delay_us(1);
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

void measure_task(void *pvParameters) {
  while(1) {
    distance = hcsr04_measure();
    ESP_LOGI("HCSR04", "Distance: %.2f cm", distance);
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}
#endif