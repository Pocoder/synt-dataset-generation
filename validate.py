import sys
import os
sys.path.append(".")

import cv2
from src.dope.inference.cuboid import Cuboid3d
from src.dope.inference.cuboid_pnp_solver import CuboidPNPSolver
from src.dope.inference.detector import ModelData, ObjectDetector
import numpy as np


def main():
    delim = 2
    fx = 1066.6690065469104 / delim
    fy = 1066.6690065469104 / delim
    width = 1920.0 / delim
    height = 1080.0 / delim
    # fx = fy = 768.16058349609375
    # width = 960
    # height = 540
    cx = width / 2
    cy = height / 2
    models = {}
    pnp_solvers = {}

    config_detect = lambda: None
    config_detect.mask_edges = 1
    config_detect.mask_faces = 1
    config_detect.vertex = 1
    config_detect.threshold = 0.5
    config_detect.softmax = 1000
    config_detect.thresh_angle = 0.5
    config_detect.thresh_map = 0.01
    config_detect.sigma = 3
    config_detect.thresh_points = 0.1

    weights = {
        # "soup": "/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/train_soup_05_19_2022/net_container_6.pth",
        "container": "/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/train_container_003_half_resolution/net_container_200.pth"
    }

    dimensions = {
        "container": [0.522630, 0.700324, 0.303825],
        "soup": [6.7659378051757813, 10.185500144958496, 6.771425724029541],
    }

    draw_colors = {
        "container": (232, 222, 12),
        "soup": (232, 222, 12),
    }

    # For each object to detect, load network model, create PNP solver, and start ROS publishers
    for model, weights_url in weights.items():
        models[model] = \
            ModelData(
                model,
                weights_url
            )
        models[model].load_net_model()

        pnp_solvers[model] = \
            CuboidPNPSolver(
                model,
                cuboid3d=Cuboid3d(dimensions[model])
            )

        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0., 1.]])
        dist_coeffs = np.array([[0.],
                        [0.],
                        [0.],
                        [0.]])

    # read the image(jpg) on which the network should be tested. 
    # example: 
    # C:\\Users\\m\\Desktop\\000044.jpg

    path_to_test = "/Users/pocoder/Downloads/2022_05_28/rendered_dataset/test"
    output_path = "/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/scripts/annotated"

    for model in models:
        # Resize camera matrix
        pnp_solvers[model].set_camera_intrinsic_matrix(camera_matrix)
        pnp_solvers[model].set_dist_coeffs(dist_coeffs)

    for root, dirs, files in os.walk(path_to_test):
        for file in files:
            if file.endswith('.json'):
                continue
            frame = cv2.imread(f'{path_to_test}/{file}')
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            if frame is None:
                continue
            height, width = frame.shape[:2]
            scaling_factor = float(400) / height
            img = np.clip(frame, 0, 255).astype(np.uint8)
            if scaling_factor < 1.0:
                img = cv2.resize(frame, (int(scaling_factor * width), int(scaling_factor * height)))

            for m in models:
                # try to detect object
                results, im_belief = ObjectDetector.detect_object_in_image(models[m].net, pnp_solvers[m], img,
                                                                           config_detect, grid_belief_debug=True,
                                                                           norm_belief=True, run_sampling=True)
                cv_imageBelief = np.array(im_belief)
                print(cv_imageBelief.shape)
                imageToShow = cv2.resize(cv_imageBelief, dsize=(800, 800))
                # cv2.imwrite(f'{output_path}/annotated_{file}', imageToShow)


if __name__ == '__main__':
    main()
