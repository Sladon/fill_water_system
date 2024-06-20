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

// Provides an interface to take an action based on the output from the bottle
// detection model.

#ifndef TENSORFLOW_LITE_MICRO_BOTTLE_DETECTION_DETECTION_RESPONDER_H_
#define TENSORFLOW_LITE_MICRO_BOTTLE_DETECTION_DETECTION_RESPONDER_H_

#include "tensorflow/lite/c/common.h"

// Called every time the results of a bottle detection run are available. The
// `bottle_score` has the numerical confidence that the captured image contains
// a bottle, and `no_bottle_score` has the numerical confidence that the image
// does not contain a bottle. Typically if bottle_score > no bottle score, the
// image is considered to contain a bottle.  This threshold may be adjusted for
// particular applications.
void RespondToDetection(float bottle_score);

#endif  // TENSORFLOW_LITE_MICRO_BOTTLE_DETECTION_DETECTION_RESPONDER_H_
