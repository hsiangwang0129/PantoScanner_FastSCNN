# -----------------------------------------------------------------------------
#
# This file is part of the PantoScanner distribution on: 
# https://huggingface.co/spaces/swissrail/PantoScanner
#
# PantoScanner - Analytics and measurement capability for technical objects.
# Copyright (C) 2017-2024 Schweizerische Bundesbahnen SBB
#
# Authors (C) 2024 L. Hofstetter (lukas.hofstetter@sbb.ch)
# Authors (C) 2017 U. Gehrig (urs.gehrig@sbb.ch)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------------

import os
from glob2 import glob
import torch
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import numpy as np
from dataclasses import dataclass, field, asdict, astuple, InitVar
from typing import List, Dict, Tuple
import streamlit as st

@dataclass
class BBoxCoordinates:
    x_min: int = field(init=True, default=None)
    x_max: int = field(init=True, default=None)
    y_min: int = field(init=True, default=None)
    y_max: int = field(init=True, default=None)
    x_center: float = field(init=False, default=False)
    y_center: float = field(init=False, default=False)
    height: int = field(init=False, default=False)
    width: int = field(init=False, default=False)
    as_array: np.ndarray = field(init=False, default=False)

    def __post_init__(self):
        self.height = self.x_max - self.x_min
        self.width = self.y_max - self.y_min
        self.x_center = 0.5 * (self.x_max + self.x_min)
        self.y_center = 0.5 * (self.y_max + self.y_min)

    def shift_coordinates(self, x_shift, y_shift, x_new_low_bound=None, x_new_upp_bound=None,
                          y_new_low_bound=None, y_new_upp_bound=None):
        self.x_min = self.x_min - x_shift
        if x_new_low_bound is not None:
            self.x_min = max(self.x_min, x_new_low_bound)
        self.x_max = self.x_max - x_shift
        if x_new_upp_bound is not None:
            self.x_max = min(self.x_max, x_new_upp_bound)
        self.y_min = self.y_min - y_shift
        if y_new_low_bound is not None:
            self.y_min = max(self.y_min, y_new_low_bound)
        self.y_max = self.y_max - y_shift
        if y_new_upp_bound is not None:
            self.y_max = min(self.y_max, y_new_upp_bound)
        self.height = self.x_max - self.x_min
        self.width = self.y_max - self.y_min
        self.x_center = 0.5 * (self.x_max + self.x_min)
        self.y_center = 0.5 * (self.y_max + self.y_min)

    def check_valid(self):
        if self.width and self.height:
            return True
        else:
            return False

    def make_yolo_label_string(self, label_number, img_size, float_precision):
        final_string = str(int(label_number))
        for this_float in [self.y_center, self.x_center, self.width, self.height]:
            final_string = final_string + ' ' + '{:.{n}f}'.format(this_float/img_size, n=float_precision)
        return final_string


def evaluate_yolo(model_path: str, img_path: str, img_size: int, bound_1: Tuple[int, int], bound_2: Tuple[int, int]):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, verbose=None)  # local model
    input_img = cv.imread(img_path)
    # img_1 = inpt_img[bound_1[0]:bound_1[0] + img_size, bound_1[1]:bound_1[1] + img_size, :]
    # img_2 = inpt_img[bound_2[0]:bound_2[0] + img_size, bound_2[1]:bound_2[1] + img_size, :]
    # inpt_imgs = [img_1, img_2]
    # results = model(inpt_imgs, size=img_size)

    # result_list_images = []
    # results_1 = results.pandas().xyxy[0].reset_index()  # img1 predictions (pandas)
    # results_2 = results.pandas().xyxy[1].reset_index()  # img1 predictions (pandas)

    # for index, row in results_1.iterrows():
    #     result_list_images.append(
    #         [row['xmin'] + bound_1[1], row['xmax'] + bound_1[1], row['ymin'] + bound_1[0], row['ymax'] + bound_1[0]])
    # for index, row in results_2.iterrows():
    #     result_list_images.append(
    #         [row['xmin'] + bound_2[1], row['xmax'] + bound_2[1], row['ymin'] + bound_2[0], row['ymax'] + bound_2[0]])

    # result_arr = np.asarray(result_list_images)
    # x_min = np.min(result_arr[:, 0])
    # x_max = np.max(result_arr[:, 1])
    # y_min = np.min(result_arr[:, 2])
    # y_max = np.max(result_arr[:, 3])

    # bound_final_x = max(min(int(0.5 * (x_min + x_max - img_size)), 3208 - img_size), 0)
    # bound_final_y = max(min(int(0.5 * (y_min + y_max - img_size)), 2200 - img_size), 0)

    # img_final = inpt_img[bound_final_y:bound_final_y + img_size, bound_final_x:bound_final_x + img_size, :]
    img_final = input_img
    imgs_final = [img_final]
    results_final = model(imgs_final, size=img_size)
    bboxes_final = results_final.pandas().xyxy[0].reset_index()  # img1 predictions (pandas)

    final_bbox_list = []
    for index, row in bboxes_final.iterrows():
        this_box = BBoxCoordinates(row['ymin'] + bound_final_y, row['ymax'] + bound_final_y,
                                   row['xmin'] + bound_final_x, row['xmax'] + bound_final_x)
        final_bbox_list.append(this_box)

    return final_bbox_list
def draw_yolo_detections(image, detections, model_names):
    """
    在圖片上繪製 YOLO 偵測框。
    - image: 原始圖片 (NumPy 陣列)
    - detections: YOLO 偵測結果 (xywh 格式)
    - model_names: YOLO 模型的類別名稱
    """
    img_copy = image.copy()
    for detection in detections:
        x_center, y_center, width, height, confidence, class_id = detection
        class_id = int(class_id)  # 轉為整數類別 ID
        class_name = model_names[class_id]  # 查找類別名稱
        confidence = float(confidence)

        # 轉換中心點格式為左上角 (x_min, y_min) 和右下角 (x_max, y_max)
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # 畫出邊界框
        cv.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv.putText(img_copy, label, (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img_copy

def evaluate_yolo_2(model_loaded, img_path: str, img_size: int, bound_1: Tuple[int, int], bound_2: Tuple[int, int]):
    model = model_loaded
    input_img = cv.imread(img_path)
    
    img_final = cv.resize(input_img,(1408,1408))
    # img_final_rgb = cv.cvtColor(img_final,cv.COLOR_BGR2RGB)
    # st.image(img_final)
    



    results_final = model(img_final, size=540)
    print("results_final",results_final)

    st.subheader("YOLO Object Detection Results")

    for i, img_result in enumerate(results_final.xywh):
        img_orig = img_final
        img_with_detections = draw_yolo_detections(img_orig, img_result, results_final.names)

        # OpenCV BGR → RGB，因為 Streamlit 需要 RGB
        img_with_detections_rgb = cv.cvtColor(img_with_detections, cv.COLOR_BGR2RGB)

        # 在 Streamlit 顯示圖片
        st.image(img_with_detections_rgb, caption=f"Detected Objects in Image {i+1}",use_container_width=True)

    bboxes_final = results_final.pandas().xywh[0]  # img1 predictions (pandas)
    print("bboxes_final: ",bboxes_final)
    # final_bbox_list = []
    # for index, row in bboxes_final.iterrows():
    #     this_box = BBoxCoordinates(row['ymin'] , row['ymax'] ,
    #                                row['xmin'] , row['xmax'] )
    #     final_bbox_list.append(this_box)

    return bboxes_final




