import os
import json
import numpy as np
import pandas as pd
from itertools import cycle, islice
from scripts.test import Draw
import cv2
from PIL import Image

from scipy.spatial.transform import Rotation as R
from pathlib import Path


def project(point, camera):
    u = camera['fx'] * point[0] / point[2] + camera['cx']
    v = camera['fy'] * point[1] / point[2] + camera['cy']
    return u, v


def transform(point, quaternion, shift):
    r = R.from_quat(quaternion).as_matrix()
    point = np.asarray(point)
    shift = np.asarray(shift)
    return r @ point + shift


def transform_back(point, quaternion, shift):
    r = R.from_quat(quaternion).as_matrix()
    point = np.asarray(point)
    shift = np.asarray(shift)
    return r.T @ (point - shift)


def create_cuboid_points(obj_dimensions):
    obj_dimensions = np.asarray(obj_dimensions)

    points = [
        (0.5, -0.5, 0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (0.5, -0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (0.5, 0.5, -0.5),
    ]
    for i, point in enumerate(points):
        point = np.asarray(point)
        point *= obj_dimensions
        points[i] = point
    return points


def get_object_info(obj_location, obj_dim, obj_quat, camera_location, camera_quaternion, camera_intrinsics):
    cuboid_points = create_cuboid_points(obj_dim)
    cuboid_points = [transform(point, obj_quat, obj_location).tolist() for point in cuboid_points]
    # local_points = [transform_back(point, camera_quaternion, camera_location) for point in cuboid_points]
    projected_cuboid_points = [project(point, camera_intrinsics) for point in cuboid_points]
    projected_cuboid_centroid = project(obj_location, camera_intrinsics)

    min_projected_xs = min([point[0] for point in projected_cuboid_points])
    min_projected_ys = min([point[1] for point in projected_cuboid_points])

    max_projected_xs = max([point[0] for point in projected_cuboid_points])
    max_projected_ys = max([point[1] for point in projected_cuboid_points])

    return {
        'camera_data': {
            'location_worldframe': camera_location,
            'quaternion_xyzw_worldframe': camera_quaternion
        },
        'objects': [
            {
                'class': 'container',
                'visibility': 1,
                'location': obj_location,
                'quaternion_xyzw': obj_quat,
                'cuboid_centroid': obj_location,
                'projected_cuboid_centroid': projected_cuboid_centroid,
                'bounding_box':
                    {
                        'top_left': [min_projected_ys, min_projected_xs],
                        'bottom_right': [max_projected_ys, max_projected_xs]
                    },
                'cuboid': cuboid_points,
                'projected_cuboid': projected_cuboid_points
            }
        ]
    }


def quaternion_inverse(quternion):
    return [-quternion[0], quternion[1], quternion[2], quternion[3]]


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return [-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
             x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
             -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
             x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0]


def main():
    root_path = Path("/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/rendered_dataset/test")
    data = pd.read_csv(root_path.joinpath('data.csv'), sep=' ', index_col=False)
    number_of_scenes = 4
    n = len(data) // number_of_scenes
    render_ids = list(islice(cycle(range(n)), len(data)))
    data['render_id'] = render_ids
    camera_intrinsics = {
        'fx': 1066.6690065469104,
        'fy': 1066.6690065469104,
        'cx': 960,
        'cy': 540,
    }
    img_size = (1920, 1080)

    for index, row in data.iterrows():
        scene_id, render_id = int(row['scene_id']), int(row['render_id'])

        bin_location = [row['bin_loc_x'], row['bin_loc_y'], row['bin_loc_z']]
        bin_quaternion = [row['bin_rot_quat_1'], row['bin_rot_quat_2'], row['bin_rot_quat_3'], row['bin_rot_quat_4']]
        
        bin_dim = [row['bin_dim_x'], row['bin_dim_y'], row['bin_dim_z']]
        camera_location = [row['camera_loc_x'], row['camera_loc_y'], row['camera_loc_z']]
        camera_quaternion = [row['camera_rot_quat_1'], row['camera_rot_quat_2'], row['camera_rot_quat_3'], row['camera_rot_quat_4']]

        # надо повернуть объекты в другую систему
        bin_dim = [bin_dim[0], bin_dim[2], bin_dim[1]]  # в новую систему
        # wxyz
        fix_quat = [0, 1, 0, 0]  # +- 180 градусов по x
        fix_bin_quat = [-0.7071067690849304, 0.7071067690849304, 0.0, 0.0]  # -90 градусов по X

        camera_quaternion = quaternion_multiply(camera_quaternion, fix_quat)

        bin_quaternion = quaternion_multiply(quaternion_inverse(camera_quaternion), bin_quaternion)

        # xyzw
        camera_quaternion = [camera_quaternion[1], camera_quaternion[2], camera_quaternion[3], camera_quaternion[0]]
        bin_quaternion = [bin_quaternion[1], bin_quaternion[2], bin_quaternion[3], bin_quaternion[0]]

        bin_location = transform_back(bin_location, camera_quaternion, camera_location).tolist()

        bin_info = get_object_info(bin_location, bin_dim, bin_quaternion, camera_location, camera_quaternion, camera_intrinsics)

        bbox = bin_info["objects"][0]['bounding_box']
        if bbox['top_left'][0] < 0.1 * img_size[1] or bbox['top_left'][1] < 0.1 * img_size[0]:
            continue
        if bbox['bottom_right'][0] > 0.9 * img_size[1] or bbox['bottom_right'][1] > 0.9 * img_size[0]:
            continue

        with open(root_path.joinpath(f'scene_{scene_id}_render_{render_id}.json'), mode='w') as f_out:
            json.dump(bin_info, f_out, indent=4)

        frame = cv2.imread(
            f'/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/rendered_dataset/test/scene_{scene_id}_render_{render_id}.png')
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame_copy = frame.copy()
        im = Image.fromarray(frame_copy)
        draw = Draw(im)

        points2d = []
        for pair in bin_info["objects"][0]['projected_cuboid']:
            points2d.append(tuple(pair))
        draw.draw_cube(points2d, (232, 222, 12))

        annotated_frame = np.array(im)
        cv2.imshow('frame', annotated_frame)
        cv2.imwrite(f'/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/render_results_small/annotated_scene_{scene_id}_render_{render_id}.png', annotated_frame)


def test():
    obj_location = [ -0.88279998302459717, -5.1195998191833496, 107.14140319824219 ]
    obj_dim = [ 7.7873997688293457, 5.1192998886108398, 5.2560000419616699 ]
    obj_quat = [ -0.083499997854232788, -0.11739999800920486, -0.22560000419616699, 0.96350002288818359 ]
    camera_location = [ -50.603099822998047, 3193.57275390625, 225.56050109863281 ]
    camera_quat = [ 0.10440000146627426, 0.4699999988079071, -0.19009999930858612, 0.85559999942779541 ]
    camera_instr = {
        'fx': 768.16058349609375,
        'fy': 768.16058349609375,
        'cx': 480,
        'cy': 270,
    }
    data = get_object_info(obj_location, obj_dim, obj_quat, camera_location, camera_quat, camera_instr)['objects'][0]

    frame = cv2.imread(
        f'/Users/pocoder/Downloads/fat/single/061_foam_brick_16k/temple_4/000099.right.jpg')
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    frame_copy = frame.copy()
    im = Image.fromarray(frame_copy)
    draw = Draw(im)

    points2d = []
    for pair in data['projected_cuboid']:
        points2d.append(tuple(pair))
    draw.draw_cube(points2d, (232, 222, 12))

    annotated_frame = np.array(im)
    cv2.imshow('frame', annotated_frame)
    cv2.imwrite(
        f'/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/render_results_small/annotated_scene.png',
        annotated_frame)


def modify_data():
    path_to_data = '/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/scripts/test'
    camera_intrinsics = {
        'fx': 1066.6690065469104,
        'fy': 1066.6690065469104,
        'cx': 960,
        'cy': 540,
    }
    img_size = (1920, 1080)

    for root, dirs, files in os.walk(path_to_data):
        render_ids = []
        for file in files:
            # if file.endswith('.png'):
            #     frame = cv2.imread(f'{path_to_data}/{file}')
            #     frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            #     cv2.imwrite(f'{path_to_data}/{file}', frame)
            if not file.endswith('.json'):
                continue
            file_name = file[:-5]
            rid = file_name.rfind('_')
            render_id = int(file_name[rid+1:])
            bin_dims = [
                [0.523, 0.7, 0.24],
                [0.525, 0.7, 0.289],
                [0.525, 0.7, 0.378],
            ]
            with open(f'/Users/pocoder/Downloads/2022_05_28/rendered_dataset/test/{file}') as f:
                data = json.load(f)
            bin_data = data['objects'][0]
            camera_data = data['camera_data']

            bin_location = bin_data['location']
            bin_quaternion = bin_data['quaternion_xyzw']
            bin_dim = bin_dims[render_id // 84]
            camera_location = camera_data['location_worldframe']
            camera_quaternion = camera_data['quaternion_xyzw_worldframe']

            # возвращаем

            bin_location = transform(bin_location, camera_quaternion, camera_location)
            camera_quaternion = [camera_quaternion[3], camera_quaternion[0], camera_quaternion[1], camera_quaternion[2]]
            # надо повернуть объекты в другую систему
            bin_dim = [bin_dim[0], bin_dim[1], bin_dim[2]]  # в новую систему
            # wxyz
            fix_bin_quat = [-0.7071067690849304, 0.7071067690849304, 0.0, 0.0]  # -90 градусов по X

            bin_quaternion = quaternion_multiply(quaternion_inverse(camera_quaternion), bin_quaternion)

            # xyzw
            camera_quaternion = [camera_quaternion[1], camera_quaternion[2], camera_quaternion[3], camera_quaternion[0]]
            bin_quaternion = [bin_quaternion[1], bin_quaternion[2], bin_quaternion[3], bin_quaternion[0]]

            bin_location = transform_back(bin_location, camera_quaternion, camera_location).tolist()

            bin_info = get_object_info(bin_location, bin_dim, bin_quaternion, camera_location, camera_quaternion,
                                       camera_intrinsics)
            with open(f'{path_to_data}/{file}', mode='w') as f_out:
                json.dump(bin_info, f_out, indent=4)


def draw():
    path_to_data = '/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/scripts/test'
    for root, dirs, files in os.walk(path_to_data):
        for file in files:
            if not file.endswith('.json'):
                continue
            with open(f'/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/scripts/test/{file}') as f:
                bin_info = json.load(f)

            frame = cv2.imread(f'{path_to_data}/{file[:-5]}.png')
            frame_copy = frame.copy()
            im = Image.fromarray(frame_copy)
            draw = Draw(im)

            points2d = []
            for point in bin_info['objects'][0]['projected_cuboid']:
                points2d.append(tuple(point))
            draw.draw_cube(points2d, (100, 100, 100))

            annotated_frame = np.array(im)
            cv2.imshow('frame', annotated_frame)
            cv2.imwrite(f'{path_to_data}/annotated_{file[:-5]}.png', annotated_frame)


if __name__ == '__main__':
    # main()
    # test()
    # modify_data()
    draw()
