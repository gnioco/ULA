# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np

from lib.KalmanFilter import KalmanFilter

MARGIN = 10  # pixels
ROW_SIZE = 30  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 0)  # black


#Create KalmanFilter object KF
#KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
KF = KalmanFilter(0.1, 1, 1, 1, 1, 1)


def visualize(image, detection_result) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualized.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    # Use the orange color for high visibility.
    cv2.rectangle(image, start_point, end_point, (0, 165, 255), 3)

    # Draw label and score
    category_name = detection.categories[0].category_name
    probability = round(detection.categories[0].score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return image

def localize(detection_result) -> np.ndarray:
  
  """Gets locations of divers on the input image and return it.
  Args:
    detection_result: The list of all "Detection" entities.
  Returns:
    Array with diver locations.
  """
  diver_location_list = []
  diver_location = [0,0]
  center = None

  # function to find deepest diver
  def find_deepest_diver(tuple_list):
    if not tuple_list:
        return None  # Return None for an empty list

    min_tuple = min(tuple_list, key=lambda x: x[1])
    return min_tuple
  
  # add divers centers to the list
  for detection in detection_result.detections:    
        if detection.categories[0].category_name == "diver":     
            bbox = detection.bounding_box
            diver_C = int(bbox.origin_x + bbox.width/2), int(bbox.origin_y + bbox.height/2)
            diver_location_list.append(diver_C)
  
  # get location of the deepest detected diver
  diver_location = find_deepest_diver(diver_location_list)
  
  if diver_location is not None:
    # Predict
    (x, y) = KF.predict()
    # Update
    (x1, y1) = KF.update(diver_location)                    
    diver_location = [int(x1[0,0]), int(x1[0,1])]

  return diver_location