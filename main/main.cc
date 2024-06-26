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

void camera_task(void) {
#if CLI_ONLY_INFERENCE
  esp_cli_start();
  vTaskDelay(portMAX_DELAY);
#else
  xTaskCreate(measure_task, "measure_task", 2 * 1024, NULL, 5, NULL);
  while (true) {
    loop();
  }
#endif
}

extern "C" void app_main() {
  setup();
  xTaskCreate((TaskFunction_t)&camera_task, "camera_task", 4 * 1024, NULL, 8, NULL);
  vTaskDelete(NULL);
}
