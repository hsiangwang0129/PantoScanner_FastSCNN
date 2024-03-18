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
    inpt_img = cv.imread(img_path)
    img_1 = inpt_img[bound_1[0]:bound_1[0] + img_size, bound_1[1]:bound_1[1] + img_size, :]
    img_2 = inpt_img[bound_2[0]:bound_2[0] + img_size, bound_2[1]:bound_2[1] + img_size, :]
    inpt_imgs = [img_1, img_2]
    results = model(inpt_imgs, size=img_size)

    result_list_images = []
    results_1 = results.pandas().xyxy[0].reset_index()  # img1 predictions (pandas)
    results_2 = results.pandas().xyxy[1].reset_index()  # img1 predictions (pandas)

    for index, row in results_1.iterrows():
        result_list_images.append(
            [row['xmin'] + bound_1[1], row['xmax'] + bound_1[1], row['ymin'] + bound_1[0], row['ymax'] + bound_1[0]])
    for index, row in results_2.iterrows():
        result_list_images.append(
            [row['xmin'] + bound_2[1], row['xmax'] + bound_2[1], row['ymin'] + bound_2[0], row['ymax'] + bound_2[0]])

    result_arr = np.asarray(result_list_images)
    x_min = np.min(result_arr[:, 0])
    x_max = np.max(result_arr[:, 1])
    y_min = np.min(result_arr[:, 2])
    y_max = np.max(result_arr[:, 3])

    bound_final_x = max(min(int(0.5 * (x_min + x_max - img_size)), 3208 - img_size), 0)
    bound_final_y = max(min(int(0.5 * (y_min + y_max - img_size)), 2200 - img_size), 0)

    img_final = inpt_img[bound_final_y:bound_final_y + img_size, bound_final_x:bound_final_x + img_size, :]
    imgs_final = [img_final]
    results_final = model(imgs_final, size=img_size)
    bboxes_final = results_final.pandas().xyxy[0].reset_index()  # img1 predictions (pandas)

    final_bbox_list = []
    for index, row in bboxes_final.iterrows():
        this_box = BBoxCoordinates(row['ymin'] + bound_final_y, row['ymax'] + bound_final_y,
                                   row['xmin'] + bound_final_x, row['xmax'] + bound_final_x)
        final_bbox_list.append(this_box)

    return final_bbox_list


def evaluate_yolo_2(model_loaded, img_path: str, img_size: int, bound_1: Tuple[int, int], bound_2: Tuple[int, int]):
    model = model_loaded
    inpt_img = cv.imread(img_path)
    img_1 = inpt_img[bound_1[0]:bound_1[0] + img_size, bound_1[1]:bound_1[1] + img_size, :]
    img_2 = inpt_img[bound_2[0]:bound_2[0] + img_size, bound_2[1]:bound_2[1] + img_size, :]
    inpt_imgs = [img_1, img_2]
    results = model(inpt_imgs, size=img_size)

    result_list_images = []
    results_1 = results.pandas().xyxy[0].reset_index()  # img1 predictions (pandas)
    results_2 = results.pandas().xyxy[1].reset_index()  # img1 predictions (pandas)

    for index, row in results_1.iterrows():
        result_list_images.append(
            [row['xmin'] + bound_1[1], row['xmax'] + bound_1[1], row['ymin'] + bound_1[0], row['ymax'] + bound_1[0]])
    for index, row in results_2.iterrows():
        result_list_images.append(
            [row['xmin'] + bound_2[1], row['xmax'] + bound_2[1], row['ymin'] + bound_2[0], row['ymax'] + bound_2[0]])

    result_arr = np.asarray(result_list_images)
    x_min = np.min(result_arr[:, 0])
    x_max = np.max(result_arr[:, 1])
    y_min = np.min(result_arr[:, 2])
    y_max = np.max(result_arr[:, 3])

    bound_final_x = max(min(int(0.5 * (x_min + x_max - img_size)), 3208 - img_size), 0)
    bound_final_y = max(min(int(0.5 * (y_min + y_max - img_size)), 2200 - img_size), 0)

    img_final = inpt_img[bound_final_y:bound_final_y + img_size, bound_final_x:bound_final_x + img_size, :]
    imgs_final = [img_final]
    results_final = model(imgs_final, size=img_size)
    bboxes_final = results_final.pandas().xyxy[0].reset_index()  # img1 predictions (pandas)

    final_bbox_list = []
    for index, row in bboxes_final.iterrows():
        this_box = BBoxCoordinates(row['ymin'] + bound_final_y, row['ymax'] + bound_final_y,
                                   row['xmin'] + bound_final_x, row['xmax'] + bound_final_x)
        final_bbox_list.append(this_box)

    return final_bbox_list


#
#
# img_folder_path = 'C:/Users/hofst/PycharmProjects/ImageAnalysis_SBB/training_data/ext_label_1/images/'
# img_list = [os.path.normpath(this_path) for this_path in glob(img_folder_path + '*.png')]
# b_box_list = []
# model_path_1 = 'C:\\Users\\hofst\\PycharmProjects\\ImageAnalysis_SBB\\yolov5\\data\\richard\\yolo_training_test_1408_size_m\\weights\\best.pt'
# img_path_new = 'C:\\Users\\hofst\\PycharmProjects\\ImageAnalysis_SBB\\training_data\\test_1.png'
# bound_1_new = (300, 92)
# bound_2_new = (650, 1500)
# img_size_new = 1408
#
# this_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path_1, verbose=False)  # local model
#
# for img_path_new in img_list:
#     test_result = evaluate_yolo_2(this_model, img_path_new, img_size_new, bound_1_new, bound_2_new)
#     b_box_list.append(test_result)
# matplotlib.use('TkAgg') # Change backend after loading model
# fig, ax = plt.subplots()
# # Display the image
# ax.imshow(cv.imread(img_path_new), cmap='gray')
# # Create a Rectangle patch
# rect = patches.Rectangle((test_result[0].y_min, test_result[0].x_min), test_result[0].y_max - test_result[0].y_min, test_result[0].x_max - test_result[0].x_min, linewidth=1, edgecolor='r', facecolor='none')
# rect_2 = patches.Rectangle((test_result[1].y_min, test_result[1].x_min), test_result[1].y_max - test_result[1].y_min, test_result[1].x_max - test_result[1].x_min, linewidth=1, edgecolor='b', facecolor='none')
#
# # Add the patch to the Axes
# ax.add_patch(rect)
# ax.add_patch(rect_2)
# fig.show()
#
# #x_1_list = []
# #y_1_list = []
# #x_2_list = []
# #y_2_list = []
#
# #for box_pair in b_box_list:
# #    box_1 = box_pair[0]
# #    box_2 = box_pair[1]
# #    x_1_list.append(box_1.y_min)
# #    y_1_list.append(box_1.x_max)
# #    x_1_list.append(box_2.y_min)
# #    y_1_list.append(box_2.x_max)
#
# #    x_2_list.append(box_1.y_max)
# #    y_2_list.append(box_1.x_min)
# #    x_2_list.append(box_2.y_max)
# #    y_2_list.append(box_2.x_min)
#
# #ax.scatter(x_1_list, y_1_list)
# #ax.scatter(x_2_list, y_2_list)
# #plt.imshow(cv.imread(img_path_new), cmap='gray')
# #plt.grid()

