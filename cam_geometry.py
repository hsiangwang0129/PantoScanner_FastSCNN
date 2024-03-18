import numpy as np
from numpy import linalg
import math


def obj_2_cam_coords(angles, distance, obj_coordinates, angle_order=('z', 'y', 'x')):
    d_x, d_y, d_z = distance  # distance from cam to object in camera coordinates
    x_obj, y_obj, z_obj = obj_coordinates
    d_cam2obj = np.asarray([d_x, d_y, d_z])
    p_obj = np.asarray([x_obj, y_obj, z_obj])
    rot_angles = {}
    for angle, axis in zip(angles, angle_order):
        rot_angles[axis] = angle

    rot_matrices = {'x': np.asarray([[1, 0, 0],
                                     [0, math.cos(rot_angles['x']), -math.sin(rot_angles['x'])],
                                     [0, math.sin(rot_angles['x']), math.cos(rot_angles['x'])]]),
                    'y': np.asarray([[math.cos(rot_angles['y']), 0, math.sin(rot_angles['y'])],
                                     [0, 1, 0],
                                     [-math.sin(rot_angles['y']), 0, math.cos(rot_angles['y'])]]),
                    'z': np.asarray([[math.cos(rot_angles['z']), -math.sin(rot_angles['z']), 0],
                                     [math.sin(rot_angles['z']), math.cos(rot_angles['z']), 0],
                                     [0, 0, 1]])}
    rot_matrix = np.eye(3)
    for axis in angle_order:
        rot_matrix = rot_matrix.dot(rot_matrices[axis])
    rot_vector = rot_matrix.dot(p_obj)
    p_cam = rot_vector + d_cam2obj
    return p_cam[0].item(), p_cam[1].item(), p_cam[2].item()


def line_surface_intersect(p_0_l, r_l, p_0_s, u_l, v_l):
    m_dir_matirx = np.asarray([[r_l[0], -u_l[0], -v_l[0]], [r_l[1], -u_l[1], -v_l[1]], [r_l[2], -u_l[2], -v_l[2]]], dtype='float')
    d_p_0 = np.asarray([p_0_s[0] - p_0_l[0], p_0_s[1] - p_0_l[1], p_0_s[2] - p_0_l[2]], dtype='float')
    inv_matrix = linalg.inv(m_dir_matirx)
    lin_sol = inv_matrix.dot(d_p_0)
    q = lin_sol[0].item()
    s = lin_sol[1].item()
    t = lin_sol[2].item()
    section_point = np.asarray([p_0_l[0], p_0_l[1], p_0_l[2]], dtype='float') + q * np.asarray([r_l[0], r_l[1], r_l[2]], dtype='float')
    return [(q, s, t), (section_point[0], section_point[1], section_point[2])]


def pixel2_physical(pix_coordinates, pixel_size, n_pix_x, n_pix_y):
    physical_coord_list = []
    if type(pix_coordinates) is not list:
        pix_coord_list = [pix_coordinates]
    else:
        pix_coord_list = pix_coordinates
    for x, y in pix_coord_list:
        x_p = - (x - 0.5 * n_pix_x) * pixel_size
        y_p = (y - 0.5 * n_pix_y) * pixel_size
        physical_coord_list.append((x_p, y_p))
    return physical_coord_list


def physical2_ray(phys_coordinates, focal_length):
    ray_direction_list = []
    if type(phys_coordinates) is not list:
        phys_coord_list = [phys_coordinates]
    else:
        phys_coord_list = phys_coordinates
    for x, y in phys_coord_list:
        length = (x**2 + y**2 + focal_length**2)**0.5
        x_new = x / length
        y_new = y / length
        z_new = focal_length / length
        ray_direction_list.append((x_new, y_new, z_new))
    return ray_direction_list


def pix2_object_surf(pix_coords, eul_angles, distance, focal_length, pixel_size, n_pix_x, n_pix_y,  p_0_surf, dir_1_surf, dir_2_surf, angle_order=('x', 'y', 'z')):  #xyz
    # p_0 is the vector pointing to the surface origin expressed in object coordinates
    # dir_surf are the directions of the two vector spanning the surface, expressed in object coordinates
    obj_surf_coords = []
    if type(pix_coords) is not list:
        pix_coord_list = [pix_coords]
    else:
        pix_coord_list = pix_coords
    physical_coords = pixel2_physical(pix_coord_list, pixel_size, n_pix_x, n_pix_y)
    ray_directions = physical2_ray(physical_coords, focal_length)
    ray_start_point = (0, 0, -focal_length)
    dir_1_surf_cam = obj_2_cam_coords(eul_angles, (0, 0, 0), dir_1_surf, angle_order=angle_order)
    dir_2_surf_cam = obj_2_cam_coords(eul_angles, (0, 0, 0), dir_2_surf, angle_order=angle_order)
    p_0_surf_cam = obj_2_cam_coords(eul_angles, distance, p_0_surf, angle_order=angle_order)

    for this_direction in ray_directions:
        result = line_surface_intersect(ray_start_point, this_direction, p_0_surf_cam, dir_1_surf_cam, dir_2_surf_cam)
        obj_surf_coords.append(result)
    return obj_surf_coords


#my_test = pix2_object_surf([(800, 1400)], [0, 0, 0], [0, 0, 6000], 75, 0.0045, 2200, 3208, [0, 0, 10], (1, 0, 0), (0, 1, 0))

