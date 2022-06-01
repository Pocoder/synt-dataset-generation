import math
import os
import json
import cv2
from matplotlib import pyplot as plt

import numpy as np
from scipy.spatial.transform import Rotation as R

from format_transformation import transform, project, create_cuboid_points


def transform_back(shift, quaternion):
    r = R.from_quat(quaternion).as_matrix()
    shift = np.asarray(shift)
    return r.T, -r.T @ shift


def main():
    predicted_data = {}
    with open('/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/scripts/annotated2/results.txt') as f:
        for line in f:
            data = json.loads(line)
            predicted_data[data['file']] = data
    actual_data = {}
    for root, dirs, files in os.walk('/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/scripts/test'):
        for file in files:
            if not file.endswith('.json'):
                continue
            with open(f'/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/scripts/test/{file}') as f:
                actual_data[file[:-5]+'.png'] = json.load(f)
    rs = []
    for file_name, render_info in actual_data.items():
        if file_name in predicted_data:
            actual_location = np.asarray(render_info['objects'][0]['location'])
            predicted_location = np.asarray(predicted_data[file_name]['location'])

            predicted_quat = predicted_data[file_name]['q_xyzw:']

            file_name_shrinked = file_name[:-4]
            render_id = int(file_name_shrinked[file_name_shrinked.rfind('_')+1:])
            bin_dims = [
                [0.523, 0.7, 0.24],
                [0.525, 0.7, 0.289],
                [0.525, 0.7, 0.378],
            ]
            bin_dim = bin_dims[render_id // 84]
            predicted_cuboid = create_cuboid_points(bin_dim)
            predicted_cuboid = [transform(point, predicted_quat, predicted_location).tolist() for point in predicted_cuboid]
            actual_cuboid = render_info['objects'][0]['cuboid']

            delim = 2
            fx = 1066.6690065469104 / delim
            fy = 1066.6690065469104 / delim
            width = 1920.0 / delim
            height = 1080.0 / delim
            cx = width / 2
            cy = height / 2
            camera_matrix = np.array([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0, 0., 1.]])
            dist_coeffs = np.array([[0.],
                                    [0.],
                                    [0.],
                                    [0.]])

            # rvec = np.array([predicted_quat[0], predicted_quat[1], predicted_quat[2]])
            # rvec = rvec / np.sqrt(np.sum(rvec ** 2))
            # rvec = rvec * math.acos(rvec[-1])
            # projected_points, _ = cv2.projectPoints(np.asarray([[0, 0, 0]], dtype=np.float32), rvec, predicted_location, camera_matrix, dist_coeffs)
            # projected_points = np.squeeze(projected_points)
            # predicted_quat, predicted_location = transform_back(predicted_location, quat)
            #
            # if predicted_location[2] < 0:
            #     # Get the opposite location
            #     predicted_location = -predicted_location
            #
            #     # Change the rotation by 180 degree
            #     rotate_angle = np.pi
            #     rotate_quaternion = R.from_rotvec(rotate_angle * predicted_location / np.sqrt(np.sum(predicted_location**2)))
            #     # quaternion = rotate_quaternion.cross(quaternion)

            # camera_intrinsics = {
            #     'fx': 1066.6690065469104 / 2,
            #     'fy': 1066.6690065469104 / 2,
            #     'cx': 960 / 2,
            #     'cy': 540 / 2,
            # }
            # frame = cv2.imread(f'/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/scripts/test/{file_name}')
            # p1 = project(predicted_location, camera_intrinsics)
            # p2 = project(actual_location, camera_intrinsics)
            # cv2.circle(frame, (int(p1[0]), int(p1[1])), 4, (255, 255, 255), thickness=-1)
            # cv2.circle(frame, (int(p2[0]), int(p2[1])), 4, (100, 100, 100), thickness=-1)
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)

            # 0=7 1=6 2=5 3=4 4=3 5=2 6=1 7=0

            mean_r = 0
            for i, vertex in enumerate(predicted_cuboid):
                min_dist = 10000
                for i in range(8):
                    min_dist = min(np.sqrt(np.sum((np.asarray(vertex) - np.asarray(actual_cuboid[7-i])) ** 2)), min_dist)
                mean_r += min_dist
            mean_r += np.sqrt(np.sum((predicted_location - actual_location) ** 2))
            res = mean_r/9
            rs.append(res)
            if res > 1:
                print(file_name)

    rs = sorted(rs)
    plt.hist(rs)
    plt.xlabel('среднее смещение')
    plt.ylabel('число изображений')
    plt.savefig('hist.png')
    plt.show()

    d = np.sqrt(0.525**2 + 0.7**2 + 0.3**2) / 10
    small_shift = list(filter(lambda x: x < d, rs))
    print('total number of test images: ', len(actual_data))
    print('total number of images with detected box: ', len(rs))
    print(f'shift less than 10% ({d}): ', len(small_shift))
    print('mean shift:', sum(rs)/len(rs))

    # not seen (<= 2): 7
    # corner (<=5): 9
    # simple? (6+): 30

if __name__ == '__main__':
    main()
