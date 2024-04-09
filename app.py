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

import streamlit as st
import os
import glob
import cv2
import numpy as np
from strip_measure_4_0 import prepare_networks_for_measurement, measure_strip
import plotly.express as px
import pandas as pd


IMG_BASE_DIR = 'images'
CAMERA_MATRIX = [
    [11100, 0, 1604],
    [0, 11100, 1100],
    [0, 0, 1]
]

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
IMAGE_WIDTH_SEG = 1408
IMAGE_HEIGHT_SEG = 576

path_yolo_model = os.path.join(os.getcwd(), 'app', 'detection_model.pt')
path_segmentation_model = os.path.join(os.getcwd(), 'app', 'segmentation_model.pth')


def get_image_paths(base_dir: str):
    return glob.glob(f'{os.getcwd()}/{base_dir}/*.png')


def get_num_images():
    return len(st.session_state['image_path_list'])


def increment_index(index_current: int, max_index: int, overflow=False, min_index=0):
    index_new = index_current + 1
    if index_new <= max_index:
        return index_new
    elif overflow:
        return min_index
    else:
        return index_current


def decrement_index(index_current: int, min_index, overflow=False, max_index=-1):
    index_new = index_current - 1
    if index_new >= min_index:
        return index_new
    elif overflow:
        return max_index
    else:
        return index_current


def callback_button_previous(overflow_index=True):
    new_index = decrement_index(st.session_state['image_index_current'], min_index=0,
                                overflow=overflow_index, max_index=st.session_state['num_images']-1)
    update_on_index_change(new_index)


def callback_button_next(overflow_index=True):
    new_index = increment_index(st.session_state['image_index_current'], st.session_state['num_images'],
                                overflow=overflow_index, min_index=0)
    update_on_index_change(new_index)


def update_on_index_change(new_index: int):
    st.session_state['image_index_current'] = new_index
    st.session_state['current_image_array'] = get_current_image()
    # put the current bale boundaries into the list, regardless of whether they have been stored to the database
    st.session_state['current_measurement'] = get_current_measurement()


def load_image_array(image_path: str):
    return cv2.imread(image_path)


def get_current_image():
    index_current = st.session_state['image_index_current']
    this_img_current = st.session_state['image_data_list'][index_current]
    if isinstance(this_img_current, np.ndarray):
        return this_img_current
    else:
        this_img_current = load_image_array(st.session_state['image_path_list'][index_current])
        st.session_state['image_data_list'][index_current] = this_img_current
        return this_img_current


def callback_button_measure():
    has_measurement, measurement_result = get_current_measurement()
    if has_measurement:
        display_cached_measurement_data()
    else:
        display_calculate_measurement_data()


def display_cached_measurement_data():
    st.info('Getting cached measurement', icon="ℹ️")
    display_measurement()


def display_calculate_measurement_data():
    with st.spinner('Calculating Profile Height....'):
        this_image_path = st.session_state['image_path_list'][st.session_state['image_index_current']]
        measurement_result = measure_image(this_image_path)
        update_measurements(measurement_result, st.session_state['image_index_current'])
    st.success('Measurement is done !')
    display_measurement()


def measure_image(image_path: str):
    measurement_result = measure_strip(img_path=image_path,
                                       model_yolo=st.session_state['models']['detection'],
                                       segmentation_model=st.session_state['models']['segmentation'],
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
    arr_0 = measurement_result[0]
    arr_1 = measurement_result[1]
    arr_0[:, 0] = np.abs(arr_0[:, 0])
    arr_1[:, 0] = np.abs(arr_1[:, 0])
    return arr_0, arr_1


def get_current_measurement():
    this_measurement = st.session_state['measurement_data_list'][st.session_state['image_index_current']]
    if this_measurement is not None:
        return True, this_measurement
    else:
        return False, None


def update_measurements(measurement, index_measurement):
    st.session_state['measurement_data_list'][index_measurement] = measurement


def display_measurement():
    has_measurement, measurement_data = get_current_measurement()
    if has_measurement:
        st.subheader(f'Profile Height')
        measurement_to_streamlit_chart(measurement_data[0], measurement_data[1])


def measurement_to_streamlit_chart(profile_array_1, profile_array_2):
    height_list = []
    coord_list = []
    indicator_list = []
    height_list.extend(profile_array_1[:, 0].tolist())
    coord_list.extend(profile_array_1[:, 1].tolist())
    indicator_list.extend(['Profile A' for _ in range(len(profile_array_1))])
    height_list.extend(profile_array_2[:, 0].tolist())
    coord_list.extend(profile_array_2[:, 1].tolist())
    indicator_list.extend(['Profile B' for _ in range(len(profile_array_2))])
    df = pd.DataFrame(dict(x=coord_list, y=height_list, indicator=indicator_list))
    fig = px.line(df, x='x', y='y', color='indicator', symbol="indicator")
    st.plotly_chart(fig, use_container_width=True)


if 'image_path_list' not in st.session_state:
    st.session_state['image_path_list'] = get_image_paths(IMG_BASE_DIR)

if 'num_images' not in st.session_state:
    st.session_state['num_images'] = get_num_images()

if 'image_data_list' not in st.session_state:
    st.session_state['image_data_list'] = [None for _ in range(st.session_state['num_images'])]

if 'image_index_current' not in st.session_state:
    st.session_state['image_index_current'] = 0

if 'current_image_array' not in st.session_state:
    st.session_state['current_image_array'] = get_current_image()

if 'measurement_data_list' not in st.session_state:
    st.session_state['measurement_data_list'] = [None for _ in range(st.session_state['num_images'])]

if 'current_measurement' not in st.session_state:
    st.session_state['current_measurement'] = get_current_measurement()

if 'models' not in st.session_state:
    seg_dplv3_model, yolo_nn_model = prepare_networks_for_measurement(model_yolo_path=path_yolo_model,
                                                                      model_segmentation_path=path_segmentation_model)
    st.session_state['models'] = {'segmentation': seg_dplv3_model, 'detection': yolo_nn_model}


image_emoji = '📷'
model_emoji = '⚙️'
profile_emoji = '📈'
st.title('PantoScanner')
#st.subheader(f'Source Image')
st.image(st.session_state['current_image_array'])
col1, col2, col3 = st.columns(3)
# insert prev button --> decrement image_selected_index i = min(i -= 1, 0) % or just overflow to last image
with col1:
    button_previous = st.button("previous Image", on_click=callback_button_previous, kwargs={'overflow_index': True})

with col2:
    button_measure = st.button("Measure")

with col3:
    button_next = st.button("next Image", on_click=callback_button_next, kwargs={'overflow_index': True})

if button_measure:
    callback_button_measure()
