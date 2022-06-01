import sys
import os
sys.path.append(".")

import cv2
from src.dope.inference.cuboid import Cuboid3d
from src.dope.inference.cuboid_pnp_solver import CuboidPNPSolver
from src.dope.inference.detector import ModelData, ObjectDetector
import numpy as np
import yaml
from PIL import Image, ImageDraw
import math


class Draw(object):
    """Drawing helper class to visualize the neural network output"""

    def __init__(self, im):
        """
        :param im: The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)

    def draw_line(self, point1, point2, line_color, line_width=2):
        """Draws line on image"""
        if point1 is not None and point2 is not None:
            self.draw.line([point1, point2], fill=line_color, width=line_width)

    def draw_dot(self, point, point_color, point_radius):
        """Draws dot (filled circle) on image"""
        if point is not None:
            xy = [
                point[0] - point_radius,
                point[1] - point_radius,
                point[0] + point_radius,
                point[1] + point_radius
           ]
            self.draw.ellipse(xy,
                              fill=point_color,
                              outline=point_color
                              )

    def draw_cube(self, points, color=(255, 0, 0)):
        """
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        """

        # draw front
        self.draw_line(points[0], points[1], color)
        self.draw_line(points[1], points[2], color)
        self.draw_line(points[3], points[2], color)
        self.draw_line(points[3], points[0], color)

        # draw back
        self.draw_line(points[4], points[5], color)
        self.draw_line(points[6], points[5], color)
        self.draw_line(points[6], points[7], color)
        self.draw_line(points[4], points[7], color)

        # draw sides
        self.draw_line(points[0], points[4], color)
        self.draw_line(points[7], points[3], color)
        self.draw_line(points[5], points[1], color)
        self.draw_line(points[2], points[6], color)

        # draw dots
        self.draw_dot(points[0], point_color=color, point_radius=4)
        self.draw_dot(points[1], point_color=color, point_radius=4)

        # draw x on the top
        self.draw_line(points[0], points[5], color)
        self.draw_line(points[1], points[4], color)


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
    config_detect.thresh_map = 0.0002
    config_detect.sigma = 3
    config_detect.thresh_points = 0.002

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

    camera_matrix = np.array([[fx,    0,         cx],
                              [0,      fy,      cy],
                              [0,      0.,        1.]])
    dist_coeffs = np.zeros((4, 1))

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

    path_to_test = "/Users/pocoder/Downloads/2022_05_28/rendered_dataset/test"
    output_path = "/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/scripts/annotated"
    # path_to_test = "/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/train_soup_05_19_2022"
    # output_path = "/Users/pocoder/Downloads/2022_05_28/Deep_Object_Pose/train_soup_05_19_2022"

    scaling_factor = float(400) / height
    if scaling_factor < 1.0:
        camera_matrix *= scaling_factor
    
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
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            if scaling_factor < 1.0:
                frame = cv2.resize(frame, (int(scaling_factor * width), int(scaling_factor * height)))

            frame_copy = frame.copy()
            im = Image.fromarray(frame_copy)
            draw = Draw(im)

            for m in models:
                # try to detect object
                results = ObjectDetector.detect_object_in_image(models[m].net, pnp_solvers[m], frame, config_detect)

                for i_r, result in enumerate(results):
                    if None not in result['projected_points']:
                        points2d = []
                        for pair in result['projected_points']:
                            points2d.append(tuple(pair))
                        draw.draw_cube(points2d, draw_colors[m])

                annotated_frame = np.array(im)
                cv2.imshow('frame', annotated_frame)
                cv2.imwrite(f'{output_path}/annotated_{file}', annotated_frame)


if __name__ == '__main__':
    main()
