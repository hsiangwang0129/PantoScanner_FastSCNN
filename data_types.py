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
from datetime import datetime
import math
import numpy as np
from dataclasses import dataclass, field, asdict, astuple, InitVar, fields, is_dataclass, _is_classvar, _is_dataclass_instance
from typing import List, Dict, Tuple
import cv2 as cv
import json
import base64


@dataclass
class RailNumber:
    hex_data: str = field(init=True, default=False)
    company_ref: int = field(init=False, default=False)
    direction: int = field(init=False, default=False)
    vehicle_number: str = field(init=False, default=False)
    vehicle_group: int = field(init=False, default=False)
    country_id: int = field(init=False, default=False)
    serial_number: int = field(init=False, default=False)
    fleet_id: int = field(init=False, default=False)
    fleet_member_id: int = field(init=False, default=False)
    check_digit: int = field(init=False, default=False)
    decoding_flag: int = field(init=False, default=False)  # indicating what kind of decoding error occured, no err = 0
    digits_vehicle_number: str = field(init=False, default=False)
    is_valid: bool = field(init=False, default=False)

    def __post_init__(self):
        return 0

    def check_if_valid(self) -> bool:
        check_sum = 0
        for index, digit in enumerate(self.digits_vehicle_number):
            multiplier = int(((-1) ** index + 3) * 0.5)  # take digits with odd place with weight 2, even placed with 1
            weighted_value = multiplier * int(digit)
            check_sum = check_sum + (weighted_value % 10) + int(math.floor(weighted_value / 10))
        expected_check_digit = int(10 * math.ceil(check_sum / 10)) - check_sum
        return expected_check_digit == self.check_digit


@dataclass
class RfidRaw:
    sys_time_stamp: float = field(init=True, default=False)
    read_data: bytes = field(init=True, default=False)


@dataclass
class RfidData:
    sys_time_stamp: float = field(init=True, default=False)
    reader_time_stamp: int = field(init=True, default=False)
    rssi: int = field(init=True, default=False)
    frequency: int = field(init=True, default=False)
    tag_phase: int = field(init=True, default=False)
    hex_data: str = field(init=True, default=False)


@dataclass
class RfidEvalHex:
    sys_time_stamp: float = field(init=True, default=False)
    sys_time_span_read: float = field(init=True, default=False)
    reader_time_stamp: float = field(init=True, default=False)
    num_reads: int = field(init=True, default=False)
    rssi_min: float = field(init=True, default=False)
    rssi_mean: float = field(init=True, default=False)
    rssi_max: float = field(init=True, default=False)
    hex_data: str = field(init=True, default=False)


@dataclass
class RfidEval:
    year: int = field(init=True, default=False)
    month: int = field(init=True, default=False)
    day: int = field(init=True, default=False)
    hour: int = field(init=True, default=False)
    minute: int = field(init=True, default=False)
    second: int = field(init=True, default=False)
    millisecond: int = field(init=True, default=False)

    sys_time_stamp: float = field(init=True, default=False)
    sys_time_span_read: float = field(init=True, default=False)
    reader_time_stamp: float = field(init=True, default=False)
    num_reads: int = field(init=True, default=False)
    rssi_min: float = field(init=True, default=False)
    rssi_mean: float = field(init=True, default=False)
    rssi_max: float = field(init=True, default=False)

    company_ref: int = field(init=True, default=False)
    direction: int = field(init=True, default=False)
    vehicle_number: str = field(init=True, default=False)
    vehicle_group: int = field(init=True, default=False)
    country_id: int = field(init=True, default=False)
    fleet_id: int = field(init=True, default=False)

    rfid_eval_hex: InitVar[RfidEvalHex] = field(init=True, default=False)
    rail_number: InitVar[RailNumber] = field(init=True, default=False)

    def __post_init__(self, rfid_eval_hex: RfidEvalHex, rail_number: RailNumber):
        if rfid_eval_hex:
            self.sys_time_stamp = rfid_eval_hex.sys_time_stamp
            self.sys_time_span_read = rfid_eval_hex.sys_time_span_read
            self.reader_time_stamp = rfid_eval_hex.reader_time_stamp
            self.num_reads = rfid_eval_hex.num_reads
            self.rssi_min = rfid_eval_hex.rssi_min
            self.rssi_mean = rfid_eval_hex.rssi_mean
            self.rssi_max = rfid_eval_hex.rssi_max
        if rail_number:
            self.company_ref = rail_number.company_ref
            self.direction = rail_number.direction
            self.vehicle_number = rail_number.vehicle_number
            self.vehicle_group = rail_number.vehicle_group
            self.country_id = rail_number.country_id
            self.fleet_id = rail_number.fleet_id
        if self.sys_time_stamp:
            dt_object = datetime.fromtimestamp(self.sys_time_stamp)
            self.year = dt_object.year
            self.month = dt_object.month
            self.day = dt_object.day
            self.hour = dt_object.hour
            self.minute = dt_object.minute
            self.second = dt_object.second
            self.millisecond = int(dt_object.microsecond / 1000)


@dataclass
class RfidEvalFault:
    year: int = field(init=False, default=False)
    month: int = field(init=False, default=False)
    day: int = field(init=False, default=False)
    hour: int = field(init=False, default=False)
    minute: int = field(init=False, default=False)
    second: int = field(init=False, default=False)
    millisecond: int = field(init=False, default=False)

    sys_time_stamp: float = field(init=False, default=False)
    sys_time_span_read: float = field(init=False, default=False)
    reader_time_stamp: float = field(init=False, default=False)
    num_reads: int = field(init=False, default=False)
    rssi_min: float = field(init=False, default=False)
    rssi_mean: float = field(init=False, default=False)
    rssi_max: float = field(init=False, default=False)
    hex_data: str = field(init=False, default=False)
    decoding_flag: int = field(init=False, default=False)
    company_ref: int = field(init=False, default=False)
    direction: int = field(init=False, default=False)
    vehicle_number: str = field(init=False, default=False)
    vehicle_number_is_valid: bool = field(init=False, default=False)

    rfid_eval_hex: InitVar[RfidEvalHex] = field(init=True, default=False)
    rail_number: InitVar[RailNumber] = field(init=True, default=False)

    def __post_init__(self, rfid_eval_hex: RfidEvalHex, rail_number: RailNumber):
        if rfid_eval_hex:
            self.sys_time_stamp = rfid_eval_hex.sys_time_stamp
            self.sys_time_span_read = rfid_eval_hex.sys_time_span_read
            self.reader_time_stamp = rfid_eval_hex.reader_time_stamp
            self.num_reads = rfid_eval_hex.num_reads
            self.rssi_min = rfid_eval_hex.rssi_min
            self.rssi_mean = rfid_eval_hex.rssi_mean
            self.rssi_max = rfid_eval_hex.rssi_max
            self.hex_data = rfid_eval_hex.hex_data
        if rail_number:
            self.company_ref = rail_number.company_ref
            self.direction = rail_number.direction
            self.vehicle_number = rail_number.vehicle_number
            self.decoding_flag = rail_number.decoding_flag
            self.vehicle_number_is_valid = rail_number.is_valid
        if self.sys_time_stamp:
            dt_object = datetime.fromtimestamp(self.sys_time_stamp)
            self.year = dt_object.year
            self.month = dt_object.month
            self.day = dt_object.day
            self.hour = dt_object.hour
            self.minute = dt_object.minute
            self.second = dt_object.second
            self.millisecond = int(dt_object.microsecond / 1000)


@dataclass
class CamSourceData:
    time_stamp: float = field(init=True, default=False)
    gain: float = field(init=True, default=False)
    exposure_time: float = field(init=True, default=False)
    conversion_gain: str = field(init=True, default=False)
    gamma: float = field(init=True, default=False)
    img_path_name: str = field(init=True, default=False)
    json_path_name: str = field(init=True, default=False)


@dataclass
class ImageMatchedData:
    year: int = field(init=True, default=False)
    month: int = field(init=True, default=False)
    day: int = field(init=True, default=False)
    hour: int = field(init=True, default=False)
    minute: int = field(init=True, default=False)
    second: int = field(init=True, default=False)
    millisecond: int = field(init=True, default=False)
    time_stamp_image: float = field(init=True, default=False)

    time_stamp_rfid: float = field(init=True, default=False)
    company_ref: int = field(init=True, default=False)
    vehicle_number: str = field(init=True, default=False)
    vehicle_group: int = field(init=True, default=False)
    country_id: int = field(init=True, default=False)
    fleet_id: int = field(init=True, default=False)
    direction: int = field(init=True, default=False)
    img_name: str = field(init=True, default=False)

    cam_source_data: InitVar[CamSourceData] = field(init=True, default=False)
    rfid_eval: InitVar[RfidEval] = field(init=True, default=False)

    def __post_init__(self, cam_source_data: CamSourceData, rfid_eval: RfidEval):
        if cam_source_data:
            self.time_stamp_image = cam_source_data.time_stamp
            dt_object = datetime.fromtimestamp(self.time_stamp_image)
            self.year = dt_object.year
            self.month = dt_object.month
            self.day = dt_object.day
            self.hour = dt_object.hour
            self.minute = dt_object.minute
            self.second = dt_object.second
            self.millisecond = int(dt_object.microsecond / 1000)
            self.img_name = cam_source_data.img_path_name

        if rfid_eval:
            self.time_stamp_rfid = rfid_eval.sys_time_stamp
            self.company_ref = rfid_eval.company_ref
            self.vehicle_number = rfid_eval.vehicle_number
            self.vehicle_group = rfid_eval.vehicle_group
            self.country_id = rfid_eval.country_id
            self.fleet_id = rfid_eval.fleet_id
            self.direction = rfid_eval.direction


@dataclass
class BBoxCoordinates:
    x_min: int = field(init=True, default=False)
    x_max: int = field(init=True, default=False)
    y_min: int = field(init=True, default=False)
    y_max: int = field(init=True, default=False)
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
    estimated_euler_angles: np.ndarray = field(init=True, default_factory=lambda: np.zeros((3, 1)))
    estimated_distances: np.ndarray = field(init=True, default_factory=lambda: np.zeros((3, 1)))
    profile_a: np.ndarray = field(init=True, default_factory=lambda: np.zeros((601, 2)))
    profile_b: np.ndarray = field(init=True, default_factory=lambda: np.zeros((601, 2)))

    sliding_strip_type: str = field(init=True, default=False)

    img_matched_source_data: InitVar[ImageMatchedData] = field(init=True, default=False)

    def __post_init__(self, img_matched_source_data: ImageMatchedData):
        if img_matched_source_data:
            self.year = img_matched_source_data.year
            self.month = img_matched_source_data.month
            self.day = img_matched_source_data.day
            self.hour = img_matched_source_data.hour
            self.minute = img_matched_source_data.minute
            self.second = img_matched_source_data.second
            self.millisecond = img_matched_source_data.millisecond
            self.img_name = os.path.basename(img_matched_source_data.img_name)
            self.time_stamp_rfid = img_matched_source_data.time_stamp_rfid
            self.company_ref = img_matched_source_data.company_ref
            self.vehicle_number = img_matched_source_data.vehicle_number
            self.vehicle_group = img_matched_source_data.vehicle_group
            self.country_id = img_matched_source_data.country_id
            self.fleet_id = img_matched_source_data.fleet_id
            self.direction = img_matched_source_data.direction


@dataclass
class ImageMeasuredData:
    year: int = field(init=True, default=False)
    month: int = field(init=True, default=False)
    day: int = field(init=True, default=False)
    hour: int = field(init=True, default=False)
    minute: int = field(init=True, default=False)
    second: int = field(init=True, default=False)
    millisecond: int = field(init=True, default=False)
    time_stamp_image: float = field(init=True, default=False)

    time_stamp_rfid: float = field(init=True, default=False)
    company_ref: int = field(init=True, default=False)
    vehicle_number: str = field(init=True, default=False)
    vehicle_group: int = field(init=True, default=False)
    country_id: int = field(init=True, default=False)
    fleet_id: int = field(init=True, default=False)
    direction: int = field(init=True, default=False)
    img_name: str = field(init=True, default=False)
    measurement_obj_path: str = field(init=True, default=False)
    measurement_txt_path: str = field(init=True, default=False)

    @classmethod
    def from_image_matched_data(cls, img_matched_data: ImageMatchedData, measurement_obj_path: str, measurement_txt_path: str):
        new_dict = asdict(img_matched_data)
        new_dict['measurement_obj_path'] = measurement_obj_path
        new_dict['measurement_txt_path'] = measurement_txt_path
        return cls(**new_dict)


@dataclass
class ImgSyncedData:
    img_name: str = field(init=True, default=False)
    img_name_remote: str = field(init=True, default=False)
    measurement_txt_path: str = field(init=True, default=False)
    measurement_txt_name_remote: str = field(init=True, default=False)


@dataclass
class ImageSourceData:
    img_file_path: str = field(init=True)
    json_file_path: str = field(init=True)
    img_array: np.ndarray = field(init=False)
    meta_data: dict = field(init=False)

    def load_source_data(self):
        self.img_array = cv.imread(self.img_file_path, cv.IMREAD_ANYDEPTH)
        with open(self.json_file_path) as this_json_file:
            self.meta_data = json.load(this_json_file)

    def __post_init__(self):
        file_path_tuple = (self.img_file_path, self.json_file_path)


@dataclass
class ApiCamSourceData:  # data type for source data storage, not public, just image data no matching/rfid
    # Data related to image file, required for I/O operations and logistical stuff
    image_base_name: str  # serves as identifier for storage & human readability, contains .file_ending
    image_unique_id: str  # hashed property or similar, has to be enforced to be globally unique
    image_file_type: str  # redundant & expected to stay .png, however for compatibility reasons this will be included
    image_file_size: int  # given in bytes in the stored version on disk, may differ if encoding/compression changes !
    image_file_name_path: str  # the absolute path where image is stored
    image_json_file_name_path: str  # the absolute path where image meta data is stored as json file or similar

    image_data_encoding: str  # how the byte-string is to be read and decoded, like 'base64', 'uint16' or 'utf-8'
    image_size_bytes: int  # specifying the size of the image_data in bytes, important for data transmission
    image_data: str  # the encoded image data as string
    image_json_dict: dict  # the data in the adjacent json file as dictionary

    # Data related to the optical result & parameters of the image
    image_width: int  # for decoding robustness & data validation, important for image analysis
    image_height: int  # for decoding robustness & data validation, important for image analysis
    image_pixel_format: str  # expected to stay 'Mono16', could become 'Mono12', 'Mono8' or sth different however
    image_exposure_time: float  # given in nanoseconds, might be obsolete
    image_analog_gain: float  # given in decibels, might be obsolete
    image_flash_wave_length: float  # given in nanometers, might be obsolete

    # Data related to camera which produced image, needed for image analysis & traceability
    camera_focal_length: float  # [mm] contains information on how to convert pixels into distances
    camera_pixel_size_width: float  # [mm] contains information on how to convert pixels into distances
    camera_pixel_size_height: float  # [mm] contains information on how to convert pixels into distances
    camera_optical_center_width: int  # given in pixels referenced to current left border, usually 1/2 of height
    camera_optical_center_height: int  # given in pixels referenced to current top border, usually 1/2 of width

    # Data related to the time & location of the taken image
    time_stamp_utc: float  # standard utc time_stamp
    time_zone: str  # important in order to convert utc timestamp correctly into date time formats
    location_latitude: float  # not relevant now, but will be in the future, inherited from camera reference
    location_longitude: float  # not relevant now, but will be in the future, inherited from camera reference
    location_track_ref: int  # not relevant now, but will be in the future, inherited from camera reference
    camera_machine_ref: str  # unique camera identifier, revealing model, firmware, etc.

    def convert_to_json_format(self):
        with open(self.image_file_name_path, "rb") as file_buffer:
            self.image_data = base64.b64encode(file_buffer.read()).decode()
        self.image_size_bytes = len(self.image_data)
        self.image_data_encoding = 'base64'
        with open(self.image_json_file_name_path) as json_file:
            self.image_json_dict = json.load(json_file)

    def save_data_from_json(self):
        img_data_as_bytes = base64.b64decode(self.image_data.encode())
        with open(self.image_file_name_path, 'wb') as image_file:
            image_file.write(img_data_as_bytes)
        with open(self.image_json_file_name_path, "w") as json_file:
            json.dump(self.image_json_dict, json_file)


@dataclass  # this is now basically the same class as ApiCamSourceData
class ApiPrivatePictureData:  # data type for source data storage, not public, just image data no matching/rfid
    # Data related to image file, required for I/O operations and logistical stuff
    image_base_name: str  # serves as identifier for storage & human readability, contains .file_ending
    image_unique_id: str  # hashed property or similar, has to be enforced to be globally unique
    image_file_type: str  # redundant & expected to stay .png, however for compatibility reasons this will be included
    image_file_size: int  # given in bytes in the stored version on disk, may differ if encoding/compression changes !
    image_file_name_path: str  # the absolute path where image is stored
    image_json_file_name_path: str  # the absolute path where image meta data is stored as json file or similar

    # Data related to the optical result & parameters of the image
    image_width: int  # for decoding robustness & data validation, important for image analysis
    image_height: int  # for decoding robustness & data validation, important for image analysis
    image_pixel_format: str  # expected to stay 'Mono16', could become 'Mono12', 'Mono8' or sth different however
    image_exposure_time: float  # given in nanoseconds, might be obsolete
    image_analog_gain: float  # given in decibels, might be obsolete
    image_flash_wave_length: float  # given in nanometers, might be obsolete

    # Data related to camera which produced image, needed for image analysis & traceability
    camera_focal_length: float  # [mm] contains information on how to convert pixels into distances
    camera_pixel_size_width: float  # [mm] contains information on how to convert pixels into distances
    camera_pixel_size_height: float  # [mm] contains information on how to convert pixels into distances
    camera_optical_center_width: int  # given in pixels referenced to current left border, usually 1/2 of height
    camera_optical_center_height: int  # given in pixels referenced to current top border, usually 1/2 of width

    # Data related to the time & location of the taken image
    time_stamp_utc: int  # standard utc time_stamp
    time_zone: str  # important in order to convert utc timestamp correctly into date time formats
    location_latitude: float  # not relevant now, but will be in the future, inherited from camera reference
    location_longitude: float  # not relevant now, but will be in the future, inherited from camera reference
    location_track_ref: int  # not relevant now, but will be in the future, inherited from camera reference
    camera_machine_ref: str  # unique camera identifier, revealing model, firmware, etc.


@dataclass
class ApiClientPictureData:  # data given to clients, attributes inherited from source data
    # Data related to image format, encoding and decoding
    image_base_name: str  # serves as unique identifier for all images, contains .file_ending
    image_file_type: str  # redundant & expected to stay .png, however for compatibility reasons this will be included
    image_width: int  # for decoding robustness & data validation, important for image analysis
    image_height: int  # for decoding robustness & data validation, important for image analysis
    image_pixel_format: str  # expected to stay 'Mono16', could become 'Mono12', 'Mono8' or sth different however
    image_data_encoding: str  # how the byte-string is to be read and decoded, like 'base64', 'uint16' or 'utf-8'
    image_size_bytes: int  # specifying the size of the image_data in bytes, important for data transmission
    image_data: str  # the encoded image data as string

    # Data related to the time stamp of the image, has duplicated information for convenience
    time_stamp_utc: int  # standard utc time_stamp
    time_zone: str  # important in order to convert utc timestamp correctly into date time formats
    year: int  # redundant for convenience
    month: int  # redundant for convenience
    day: int  # redundant for convenience
    hour: int  # redundant for convenience
    minute: int  # redundant for convenience
    second: int  # redundant for convenience

    # RFID-Tag related data, maybe include additional rfid-tags read in a close time window revealing train composition
    has_matched_rfid_tag: bool  # for database reasons and clarifying that the sent rfid data below is valid or not
    company_ref: int  # from epc-tag, may not always indicate the actual owner of the vehicle, important for filtering
    vehicle_number: str  # 12 digit european rail asset number, carrying all important vehicle data
    vehicle_group: int  # redundant for convenience
    country_id: int  # redundant for convenience
    vehicle_number_national: int  # redundant for convenience
    fleet_id: int  # redundant for convenience
    direction_flag: int  # can be used to get a better understanding of the pantograph orientation and location

    # Data related to the location of the picture taken, possibly direction of movement of the train
    image_location_latitude: float  # not relevant now, but will be in the future, inherited from camera reference
    image_location_longitude: float  # not relevant now, but will be in the future, inherited from camera reference
    image_location_track_ref: int  # not relevant now, but will be in the future, inherited from camera reference
    image_train_direction_of_movement: float  # train heading in degrees, or +1/-1 if the given track defines direction


