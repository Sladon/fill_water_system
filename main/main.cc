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
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_main.h"

#if CLI_ONLY_INFERENCE
#include "esp_cli.h"
#endif

void tf_main(void) {
  setup();
#if CLI_ONLY_INFERENCE
  esp_cli_start();
  vTaskDelay(portMAX_DELAY);
#else
  while (true) {
    loop();
  }
#endif
}

extern "C" void app_main() {
  hardware_init();
  xTaskCreate(hcsr04_task, "hcsr04_task", 2 * 1024, NULL, 5, NULL);
  xTaskCreate((TaskFunction_t)&tf_main, "tf_main", 4 * 1024, NULL, 8, NULL);
  vTaskDelete(NULL);
}


// #include "main_functions.h"
// #include "esp_log.h"
// #include "esp_system.h"
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "driver/gpio.h"
// #include "esp_timer.h"

// /* Define pins for the HC-SR04 sensor */
// #define RELAY_PIN GPIO_NUM_16
// #define TRIGGER_PIN GPIO_NUM_17
// #define ECHO_PIN GPIO_NUM_18
// #define TRIGGER_HIGH_TIME_US 10

// /* Define constants for calculations */
// #define SPEED_OF_SOUND_CM_PER_MICROSECOND 0.0343
// #define MICROSECONDS_TO_CM_DIVISOR 2
// #define PULSE_DURATION_MICROSECONDS 10
// #define TRIG_PIN_DELAY_MICROSECONDS 2

// /* Here you might have other includes and setup related to TensorFlow Lite */
// static const char *TAG = "HC-SR04";
// portMUX_TYPE mySpinlock;
// void hcsr04_init() {
//     gpio_config_t io_conf;
    
//     // Configure trigger pin as output
//     io_conf.intr_type = GPIO_INTR_DISABLE;
//     io_conf.mode = GPIO_MODE_OUTPUT;
//     io_conf.pin_bit_mask = (1ULL << TRIGGER_PIN);
//     io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
//     io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
//     gpio_config(&io_conf);
    
//     // Configure echo pin as input
//     io_conf.intr_type = GPIO_INTR_DISABLE;
//     io_conf.mode = GPIO_MODE_INPUT;
//     io_conf.pin_bit_mask = (1ULL << ECHO_PIN);
//     io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
//     io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
//     gpio_config(&io_conf);
//     spinlock_initialize(&mySpinlock);
// }

// float hcsr04_measure() {    
//     // Send a 10us pulse to trigger
//     gpio_set_level(TRIGGER_PIN, 0);
//     esp_rom_delay_us(2);
//     portENTER_CRITICAL(&mySpinlock);
//       gpio_set_level(TRIGGER_PIN, 1);
//       esp_rom_delay_us(TRIGGER_HIGH_TIME_US);
//       gpio_set_level(TRIGGER_PIN, 0);
//     portEXIT_CRITICAL(&mySpinlock);
    
//     // Wait for echo to be high
//     int timeout = 3000;  // Maximum waiting time in microseconds
//     while (gpio_get_level(ECHO_PIN) == 0 && timeout > 0) {
//         esp_rom_delay_us(1);  // Small delay to prevent overloading CPU
//         taskYIELD();      // Yield to reset WDT and allow other tasks to run
//         --timeout;
//     }
//     int64_t start_time = esp_timer_get_time();

//     if (timeout == 0) {
//         ESP_LOGW("DistanceTask", "Timeout waiting for ECHO_PIN to go high");
//         return -1;
//     }

//     // Wait for ECHO_PIN to go low (end of the echo pulse)
//     timeout = 3000;
//     while (gpio_get_level(ECHO_PIN) == 1 && timeout > 0) {
//         esp_rom_delay_us(1);  // Small delay to prevent overloading CPU
//         taskYIELD();      // Yield to reset WDT and allow other tasks to run
//         --timeout;
//     }
//     int64_t end_time = esp_timer_get_time();

//     if (timeout == 0) {
//         ESP_LOGW("DistanceTask", "Timeout waiting for ECHO_PIN to go low");
//         return -1;
//     }

//     // Calculate the duration of the pulse
//     long duration = end_time - start_time;
    
//     // Calculate distance in cm (speed of sound is 34300 cm/s)
//     float distance = (duration / MICROSECONDS_TO_CM_DIVISOR) * SPEED_OF_SOUND_CM_PER_MICROSECOND;
    
//     return distance;
// }

// void hcsr04_task(void *pvParameters) {
//     while(1) {
//         float distance = hcsr04_measure();
//         ESP_LOGI(TAG, "Distance: %.2f cm", distance);
//         vTaskDelay(pdMS_TO_TICKS(1000)); // Delay 1 second
//     }
// }

// extern "C" void app_main() {
//     hcsr04_init();
//     xTaskCreate(hcsr04_task, "hcsr04_task", 2048, NULL, 5, NULL);
    
    // xTaskCreate(&RelayControlTask, "RelayControlTask", 2048, NULL, 5, NULL);

    // /* Your existing bottle detection code and task creation here */
    // xTaskCreate((TaskFunction_t)&tf_main, "tf_main", 4 * 1024, NULL, 8, NULL);

//     vTaskDelete(NULL);
// }

/* Other code related to TensorFlow Lite setup and inference */
