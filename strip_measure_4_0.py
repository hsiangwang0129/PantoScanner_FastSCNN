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

import cv2 as cv
import numpy as np
import math
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from torch.cuda.amp import autocast
from object_detection_services import evaluate_yolo_2
from cam_geometry import pix2_object_surf
from scipy.spatial.transform import Rotation as R
from scipy.ndimage.morphology import binary_dilation
from data_types import BBoxCoordinates
from typing import Tuple, Union, List
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib
import pickle
import streamlit as st
import torch
from fast_scnn.fast_scnn import FastSCNN
from PIL import Image
""" Data Classes """


@dataclass
class StripMeasuredData:
    year: int = field(init=True, default=False)
    month: int = field(init=True, default=False)
    day: int = field(init=True, default=False)
    hour: int = field(init=True, default=False)
    minute: int = field(init=True, default=False)
    second: int = field(init=True, default=False)
    millisecond: int = field(init=True, default=False)
    time_stamp_image: float = field(init=True, default=False)
    img_name: str = field(init=True, default=False)

    time_stamp_rfid: float = field(init=True, default=False)
    company_ref: int = field(init=True, default=False)
    vehicle_number: str = field(init=True, default=False)
    vehicle_group: int = field(init=True, default=False)
    country_id: int = field(init=True, default=False)
    fleet_id: int = field(init=True, default=False)
    direction: int = field(init=True, default=False)

    bounding_box_a: BBoxCoordinates = field(init=True, default=False)
    bounding_box_b: BBoxCoordinates = field(init=True, default=False)
    estimated_euler_angles: np.ndarray = field(init=True, default=np.zeros((3, 1)))
    estimated_distances: np.ndarray = field(init=True, default=np.zeros((3, 1)))
    profile_a: np.ndarray = field(init=True, default=np.zeros((601, 2)))
    profile_b: np.ndarray = field(init=True, default=np.zeros((601, 2)))
    sliding_strip_type: str = field(init=True, default=False)


""" Geometry and Perspective Calculations """


def make_line(start_point, end_point, leading_coordinate=None, include_endpoint=True):
    x_start, y_start = start_point
    x_end, y_end = end_point
    if isinstance(x_start, np.ndarray):
        x_start = x_start.item()
        y_start = y_start.item()
        x_end = x_end.item()
        y_end = y_end.item()
    delta_x = x_end - x_start
    delta_y = y_end - y_start
    point_list = []

    if leading_coordinate is None:
        if abs(delta_y) > abs(delta_x):
            leading_coordinate = 'y'
        elif abs(delta_x) >= abs(delta_y):
            leading_coordinate = 'x'

    if leading_coordinate == 'y' and delta_y == 0 and delta_x != 0:
        leading_coordinate = 'x'
    if leading_coordinate == 'x' and delta_x == 0 and delta_y != 0:
        leading_coordinate = 'y'

    if leading_coordinate == 'y':
        y_values = list(range(math.floor(y_start), math.ceil(y_end) + include_endpoint, int(np.sign(delta_y).item())))
        incr_x = delta_x / abs(delta_y)
        for y in y_values:
            x = x_start + incr_x * abs(y - y_start)
            point_list.append([round(x), round(y)])
    elif leading_coordinate == 'x':
        x_values = list(range(math.floor(x_start), math.ceil(x_end) + include_endpoint, int(np.sign(delta_x).item())))
        incr_y = delta_y / abs(delta_x)
        for x in x_values:
            y = y_start + incr_y * abs(x - x_start)
            point_list.append([round(x), round(y)])

    return point_list


def calc_rect_corners(x_center, y_center, x_span, y_span, angle):
    x_1_1 = int(x_center + y_span / 2 * math.cos(angle) - x_span / 2 * math.sin(angle))
    y_1_1 = int(y_center + y_span / 2 * math.sin(angle) + x_span / 2 * math.cos(angle))

    x_1_0 = int(x_center - y_span / 2 * math.cos(angle) - x_span / 2 * math.sin(angle))
    y_1_0 = int(y_center - y_span / 2 * math.sin(angle) + x_span / 2 * math.cos(angle))

    x_0_0 = int(x_center - y_span / 2 * math.cos(angle) + x_span / 2 * math.sin(angle))
    y_0_0 = int(y_center - y_span / 2 * math.sin(angle) - x_span / 2 * math.cos(angle))

    x_0_1 = int(x_center + y_span / 2 * math.cos(angle) + x_span / 2 * math.sin(angle))
    y_0_1 = int(y_center + y_span / 2 * math.sin(angle) - x_span / 2 * math.cos(angle))

    return [[x_1_1, y_1_1], [x_1_0, y_1_0], [x_0_0, y_0_0], [x_0_1, y_0_1]]


def estimate_pose(camera_matrix, object_points, img_points):
    object_points = np.asarray(object_points, dtype='float32')
    img_points = np.asarray(img_points, dtype='float32')
    cam_matrix = np.asarray(camera_matrix, dtype='float32')
    result_1 = cv.solvePnP(object_points, img_points, cam_matrix, distCoeffs=None)
    return result_1


""" File Loading and Torch related Functions """


def prepare_networks_for_measurement(model_yolo_path: str, model_segmentation_path: str):
    if torch.cuda.is_available():
        this_device = "cuda"
        print('Using GPU')
    else:
        this_device = "cpu"
        print('Using CPU')

    # Load FastSCNN to replace DeepLabV3.
    segmentation_model = FastSCNN(num_classes=2)  
    checkpoint = torch.load(model_segmentation_path, map_location=this_device)
    segmentation_model.load_state_dict(checkpoint["model_state_dict"])  # 提取正確的 key

    segmentation_model.to(this_device).eval()

    # 加載 YOLOv5
    model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=model_yolo_path, verbose=None).to(this_device)

    return segmentation_model, model_yolo


def load_img_2input_tensor_1_channel(img_path, crop_bounds=None):
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.449], std=[0.226]), ])
    image = cv.imread(img_path, cv.IMREAD_ANYDEPTH)
    print(image.dtype)
    if image.dtype == 'uint16':
        image = np.array(image / 257).astype('uint8')
    if crop_bounds is not None:
        image = image[crop_bounds[0]:crop_bounds[1], crop_bounds[2]:crop_bounds[3]]
    height, width = np.shape(image)
    input_image = preprocess(image)
    
    # Typecasting
    input_image = input_image.type(torch.float)
    input_image = torch.reshape(input_image, (1, height, width))
    input_image = torch.unsqueeze(input_image, 0)
    if torch.cuda.is_available():
        this_device = "cuda"
    else:
        this_device = "cpu"
    input_image = input_image.to(this_device)
    return input_image


""" Mask Functions """


def make_rectangular_mask(rectangle_corner_list, mask_shape):
    """
    :param rectangle_corner_list: numpy array of the mask, indicating where the profile lines even can be
    :param mask_shape: BBoxCoordinates of the sliding strip lying in the upper left part of the image
    :return: contours upper left, contours lower right
    """
    x_max_target, y_max_target = mask_shape
    new_corner_list = rectangle_corner_list.copy()
    new_corner_list.append(rectangle_corner_list[0])
    boundary_lines = [make_line(start_corner, end_corner) for start_corner, end_corner in
                      zip(new_corner_list[0:-1], new_corner_list[1:])]
    boundary_lines = [[int(row[0]), int(row[1])] for line in boundary_lines for row in line]
    boundary_lines = np.asarray(boundary_lines)
    boundary_lines = boundary_lines[boundary_lines[:, 1].argsort()]
    unique_y_coordinates = np.unique(boundary_lines[:, 1])
    mask_x_max = max(np.max(boundary_lines[:, 0]).item() + 1, x_max_target)
    mask_y_max = max(np.max(boundary_lines[:, 1]).item() + 1, y_max_target)
    rect_mask = np.zeros((int(mask_x_max), int(mask_y_max)))

    for y in unique_y_coordinates:
        x_indexes = boundary_lines[:, 1] == y
        x_max_val = np.max(boundary_lines[x_indexes][:, 0])
        x_min_val = np.min(boundary_lines[x_indexes][:, 0])
        y_index = int(y)
        x_min_index = int(x_min_val)
        x_max_index = int(x_max_val)
        rect_mask[x_min_index:x_max_index, y_index] = 1

    rect_mask = binary_dilation(rect_mask)
    rect_mask = rect_mask[0:x_max_target, 0:y_max_target]
    return rect_mask


def generate_single_strip_boundary_masks(bbox_upper_left: BBoxCoordinates, bbox_lower_right: BBoxCoordinates,
                                         mask_shape):
    rot_angle_upper = math.atan(bbox_upper_left.height / bbox_upper_left.width) - 0.02
    rot_angle_lower = math.atan(bbox_lower_right.height / bbox_lower_right.width) - 0.02
    width_upper = ((bbox_upper_left.x_max - bbox_upper_left.x_min) ** 2 + (
                bbox_upper_left.y_max - bbox_upper_left.y_min) ** 2) ** 0.5
    width_lower = ((bbox_lower_right.x_max - bbox_lower_right.x_min) ** 2 + (
                bbox_lower_right.y_max - bbox_lower_right.y_min) ** 2) ** 0.5
    height = 150
    corners_upper = calc_rect_corners(bbox_upper_left.x_center, bbox_upper_left.y_center, width_upper, height,
                                      rot_angle_upper)
    corners_lower = calc_rect_corners(bbox_lower_right.x_center, bbox_lower_right.y_center, width_lower, height,
                                      rot_angle_lower)
    rect_mask_upper = make_rectangular_mask(corners_upper, mask_shape)
    rect_mask_lower = make_rectangular_mask(corners_lower, mask_shape)
    return rect_mask_upper, rect_mask_lower


def extract_profile_lines(mask_array: np.ndarray, bbox_upper_left: BBoxCoordinates, bbox_lower_right: BBoxCoordinates):
    """
    :param mask_array: numpy array of the mask, indicating where the profile lines even can be
    :param bbox_upper_left: BBoxCoordinates of the sliding strip lying in the upper left part of the image
    :param bbox_lower_right: BBoxCoordinates of the sliding strip lying in the lower right part of the image
    :return: contours upper left, contours lower right
    """
    shape_mask = np.shape(mask_array)
    rect_mask_upper_left, rect_mask_lower_right = generate_single_strip_boundary_masks(bbox_upper_left,
                                                                                       bbox_lower_right, shape_mask)
    final_mask_upper_left = np.multiply(rect_mask_upper_left, mask_array)
    final_mask_lower_right = np.multiply(rect_mask_lower_right, mask_array)
    contours_upper_left = mask_2_contour_lines(final_mask_upper_left)
    contours_lower_right = mask_2_contour_lines(final_mask_lower_right)
    return contours_upper_left, contours_lower_right
    # after this function the coordinates are ready for perspective transformation


def mask_2_contour_lines(img_mask):
    bin_mask_list = []
    for value in range(1, 5):
        bin_mask = img_mask == value
        bin_mask_list.append(binary_dilation(bin_mask))
    bin_1_2 = np.multiply(bin_mask_list[0], bin_mask_list[1])
    bin_2_3 = np.multiply(bin_mask_list[1], bin_mask_list[2])
    bin_3_4 = np.multiply(bin_mask_list[2], bin_mask_list[3])
    return bin_1_2, bin_2_3, bin_3_4


""" Contour Functions """


def discretize_contour(input_contour: np.ndarray, discretization_index=1, reduction='mean'):
    disc_min = math.ceil(np.min(input_contour[:, discretization_index]))
    disc_max = math.floor(np.max(input_contour[:, discretization_index]))
    disc_vals = input_contour[:, discretization_index]
    func_vals = input_contour[:, 1 - discretization_index]
    new_list = []
    for index in range(disc_min, disc_max + 1):
        l_bound = index - 0.5
        u_bound = index + 0.5
        vals = func_vals[np.logical_and(disc_vals < u_bound, l_bound < disc_vals)]
        if np.shape(vals)[0] > 0:
            if reduction == 'mean':
                new_val = np.mean(vals)
            elif reduction == 'min':
                new_val = np.min(vals)
            elif reduction == 'max':
                new_val = np.max(vals)
            else:
                new_val = np.mean(vals)
            if discretization_index == 1:
                new_list.append([new_val, index])
            elif discretization_index == 0:
                new_list.append([index, new_val])
    return np.asarray(new_list)


def correct_countours_mean(alu_lower: np.ndarray, alu_higher: np.ndarray, coal_profile: np.ndarray, n_elements=100):
    new_alu_lower = np.copy(alu_lower)
    new_alu_higher = np.copy(alu_higher)
    new_coal_profile = np.copy(coal_profile)

    mean_biggest_lower = np.mean(get_n_biggest_elements(alu_lower, n_elements, 1)[:, 1])
    mean_smallest_lower = np.mean(get_n_smallest_elements(alu_lower, n_elements, 1)[:, 1])
    mean_biggest_higher = np.mean(get_n_biggest_elements(alu_higher, n_elements, 1)[:, 1])
    mean_smallest_higher = np.mean(get_n_smallest_elements(alu_higher, n_elements, 1)[:, 1])
    shift = int(0.5 * (mean_biggest_lower + mean_smallest_lower + mean_biggest_higher + mean_smallest_higher))
    # print('mean shift value : ', shift)
    new_alu_lower[:, 1] = alu_lower[:, 1] - shift
    new_alu_higher[:, 1] = alu_higher[:, 1] - shift
    new_coal_profile[:, 1] = coal_profile[:, 1] - shift
    return new_alu_lower, new_alu_higher, new_coal_profile


def correct_shear_contours(alu_lower: np.ndarray, alu_higher: np.ndarray, coal_profile: np.ndarray, n_elements=50):
    new_alu_lower = np.copy(alu_lower)
    new_alu_higher = np.copy(alu_higher)
    new_coal_profile = np.copy(coal_profile)
    mean_y_biggest_lower = np.mean(get_n_biggest_elements(alu_lower, n_elements, 1)[:, 1])
    mean_x_biggest_lower = np.mean(get_n_biggest_elements(alu_lower, n_elements, 1)[:, 0])
    mean_y_smallest_lower = np.mean(get_n_smallest_elements(alu_lower, n_elements, 1)[:, 1])
    mean_x_smallest_lower = np.mean(get_n_smallest_elements(alu_lower, n_elements, 1)[:, 0])
    mean_y_biggest_higher = np.mean(get_n_biggest_elements(alu_higher, n_elements, 1)[:, 1])
    mean_x_biggest_higher = np.mean(get_n_biggest_elements(alu_higher, n_elements, 1)[:, 0])
    mean_y_smallest_higher = np.mean(get_n_smallest_elements(alu_higher, n_elements, 1)[:, 1])
    mean_x_smallest_higher = np.mean(get_n_smallest_elements(alu_higher, n_elements, 1)[:, 0])
    mean_y_smallest = 0.5 * (mean_y_smallest_higher + mean_y_smallest_lower)
    mean_x_smallest = 0.5 * (mean_x_smallest_higher + mean_x_smallest_lower)
    mean_y_biggest = 0.5 * (mean_y_biggest_higher + mean_y_biggest_lower)
    mean_x_biggest = 0.5 * (mean_x_biggest_higher + mean_x_biggest_lower)
    delta_y = mean_y_biggest - mean_y_smallest
    delta_x = mean_x_biggest - mean_x_smallest
    inclination = delta_x / delta_y
    # print('inclination value : ', inclination)
    new_alu_lower[:, 0] = alu_lower[:, 0] - inclination * alu_lower[:, 1]
    new_alu_higher[:, 0] = alu_higher[:, 0] - inclination * alu_higher[:, 1]
    new_coal_profile[:, 0] = coal_profile[:, 0] - inclination * coal_profile[:, 1]
    return new_alu_lower, new_alu_higher, new_coal_profile


def harmonize_disc_contours(disc_contour_1, disc_contour_2, harmonize_index=1):
    new_contour_1 = []
    new_contour_2 = []
    for row in disc_contour_1:
        equal_index = np.argwhere(disc_contour_2[:, harmonize_index] == row[harmonize_index])
        if np.shape(equal_index)[0]:
            new_contour_1.append([row[0], row[1]])
            new_contour_2.append([disc_contour_2[equal_index[0, 0], 0], row[1]])
    return np.asarray(new_contour_1), np.asarray(new_contour_2)


""" Sliding Strip related specific functions """


def fit_lower_base_new(low_base_cont_x, low_base_cont_y, lower_contour_quadratic_constant):
    quad_feat = - np.power(low_base_cont_x, 2) * lower_contour_quadratic_constant
    new_base_y = low_base_cont_y - quad_feat
    mat_a = np.vstack([low_base_cont_x, np.ones(len(low_base_cont_x))]).T
    m, c = np.linalg.lstsq(mat_a, new_base_y, rcond=None)[0]
    new_predict = quad_feat + m * low_base_cont_x + c
    return new_predict


""" Utility Functions """


def get_n_biggest_elements(inpt_array: np.ndarray, n_elements: int, dim_index):
    elements_of_interest = inpt_array[:, dim_index]
    ind = np.argpartition(elements_of_interest, -n_elements)[-n_elements:]
    top_n = inpt_array[ind, :]
    return top_n


def get_n_smallest_elements(inpt_array: np.ndarray, n_elements: int, dim_index):
    elements_of_interest = inpt_array[:, dim_index]
    ind = np.argpartition(elements_of_interest, n_elements)[:n_elements]
    top_n = inpt_array[ind, :]
    return top_n


""" Aggregative Function """


def measure_strip(img_path: str, model_yolo: torch.nn.Module, segmentation_model: smp.DeepLabV3,
                  camera_matrix: List[List[Union[int, float]]], object_reference_points: List[List[Union[int, float]]],
                  camera_parameters: Tuple[int, float, int, int],
                  plane_parameters_close: tuple, plane_parameters_far: tuple,
                  lower_contour_quadratic_constant: float, boundary_1: Tuple[int, int], boundary_2: Tuple[int, int],
                  image_size_seg: int, image_width_seg: int, image_height_seg: int) -> Tuple[np.ndarray, np.ndarray]:

    

    object_detection_result = evaluate_yolo_2(model_yolo, img_path, image_size_seg, boundary_1, boundary_2)
    print(object_detection_result["width"])
    xcenter, ycenter = int(object_detection_result["ycenter"])/1408*3024, int(object_detection_result["xcenter"])/1408*4032
    y_min = ycenter-image_height_seg//2
    x_min = xcenter-image_width_seg//2
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(img_path).convert('RGB')
    print("original size of image",image.size)
    cropped = image.crop((x_min, y_min, x_min + image_width_seg, y_min + image_height_seg))
    # print(f"cropped shape before transform: {cropped.size}")  # 查看 PIL Image 的尺寸
    target_size = (3200, 315)  
    final_img = cropped.resize(target_size, Image.BILINEAR)
    final_img = transform(final_img).unsqueeze(0)
    st.image(cropped, use_container_width=True)
    # print(f"cropped shape after transform: {cropped.shape}")  # 查看 Tensor 的尺寸


    with torch.set_grad_enabled(False):
        with autocast():
            prediction = segmentation_model.forward(final_img)
    print(prediction)
    if isinstance(prediction, tuple):  # 🔥 如果是 tuple，取第一個輸出
        prediction = prediction[0]
    # 確保 prediction 是 tensor 才能執行 argmax
    if isinstance(prediction, torch.Tensor):
        mask = prediction.argmax(1)
    else:
        raise TypeError(f"Expected prediction to be torch.Tensor, but got {type(prediction)}")
    
    
    return cropped,mask






if __name__ == '__main__':
    CAMERA_MATRIX = [
    [11100, 0, 1604],
    [0, 11100, 1100],
    [0, 0, 1]]

    OBJECT_REFERENCE_POINTS = [
        [347, 0, 42],  # B
        [347, 0, 522],  # D
        [-347, 26, 480],  # F
        [-347, 26, 0]]  # H
    
    LOWER_CONTOUR_QUADRATIC_CONSTANT = 0.00005
    CAMERA_PARAMETERS = (50, 0.0045, 2200, 3208)
    PLANE_PARAMETERS_CLOSE = ([0, 0, 0], (1, 0, 0), (0, 1, 0))  # Vector from pantograph coordinate frame to plane origin
    PLANE_PARAMETERS_FAR = ([0, 0, 480], (1, 0, 0), (0, 1, 0))  # Vector from pantograph coordinate frame to plane origin
    BOUNDARY_1 = (300, 92)
    BOUNDARY_2 = (650, 1500)
    IMAGE_SIZE_SEG = 1408
    # IMAGE_WIDTH_SEG = 1408
    IMAGE_WIDTH_SEG = 3200
    # IMAGE_HEIGHT_SEG = 576
    IMAGE_HEIGHT_SEG = 320
    
    path_image = 'testing/2022_10_02, 06_08_56_715981.png'  #'testing/2022_10_15, 06_07_50_539720.png'
    path_yolo_model = 'app/best.pt'
    path_segmentation_model = 'app/31_best_model.pth'
    seg_dplv3_model, yolo_nn_model = prepare_networks_for_measurement(model_yolo_path=path_yolo_model,
                                                                      model_segmentation_path=path_segmentation_model)

    measurement_result = measure_strip(img_path=path_image,
                                       model_yolo=yolo_nn_model,
                                       segmentation_model=seg_dplv3_model,
                                       camera_matrix=CAMERA_MATRIX,
                                       object_reference_points=OBJECT_REFERENCE_POINTS,
                                       camera_parameters=CAMERA_PARAMETERS,
                                       plane_parameters_close=PLANE_PARAMETERS_CLOSE,
                                       plane_parameters_far=PLANE_PARAMETERS_FAR,
                                       lower_contour_quadratic_constant=LOWER_CONTOUR_QUADRATIC_CONSTANT,
                                       boundary_1=BOUNDARY_1,
                                       boundary_2=BOUNDARY_2,
                                       image_size_seg=IMAGE_SIZE_SEG,
                                       image_width_seg=IMAGE_WIDTH_SEG,
                                       image_height_seg=IMAGE_HEIGHT_SEG)


    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        for index in range(1, box_pts):
            y_smooth[index] = np.mean(y[0: 2 * index])
            y_smooth[-1 - index] = np.mean(y[-1 - 2 * index:-1])
        y_smooth[0] = 0.5 * y[0] + 0.5 * y[1]
        y_smooth[-1] = 0.5 * y[-2] + 0.5 * y[-1]
        return y_smooth


    m_0_array: np.ndarray = measurement_result[0]
    m_1_array: np.ndarray = measurement_result[1]

    matplotlib.use('TkAgg')
    #plt.plot(measurement_result[0][:, 1], smooth(measurement_result[0][:, 0], 10))
    plt.plot(measurement_result[1][:, 1], -measurement_result[1][:, 0])
    plt.plot(measurement_result[0][:, 1], -measurement_result[0][:, 0])
    plt.xlim(-300, 300)
    plt.ylabel('Profilhöhe')
    plt.yticks([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    plt.grid()