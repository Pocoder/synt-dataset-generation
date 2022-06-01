import os
import bpy
import csv
import random
import math
from copy import copy
import numpy as np


def get_bin(scene_idx):
    return bpy.data.objects[f'bin_{scene_idx}']
    
def get_camera(scene_idx):
    return bpy.data.objects[f'camera_{scene_idx}']

def color_background_objects():
    object_names = ['machine', 'srchinteriors', 'Floor', 'Shape01', 'Rectangle', ]
    for object in bpy.data.objects:
        if any([object.name.startswith(name) for name in object_names]):
            for slot in object.material_slots:
                slot.material.node_tree.nodes['Principled BSDF'].inputs[0].default_value = (
                    random.uniform(0.001, 0.08),
                    random.uniform(0.001, 0.08),
                    random.uniform(0.001, 0.08),
                    1
                )

def random_color(obj):
    obj.active_material.node_tree.nodes['RGB'].outputs[0].default_value = (
        random.uniform(0.001, 0.08),
        random.uniform(0.001, 0.08),
        random.uniform(0.001, 0.08),
        1
    )

def look_at(camera, point):
    direction = point.location - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

def random_location(obj, constraints):
    x_constr, y_constr, z_constr = constraints
    new_x = random.uniform(*x_constr)
    new_y = random.uniform(*y_constr)
    new_z = random.uniform(*z_constr)
    obj.location[0] = new_x
    obj.location[1] = new_y
    obj.location[2] = new_z

def random_location_in_sphere(obj, point, constraints):
    def polar2cart(r, phi, theta):
        return [
             r * math.sin(theta) * math.cos(phi),
             r * math.sin(theta) * math.sin(phi),
             r * math.cos(theta)
        ]
    r_constr, theta_constr, phi_constr = constraints
    r = random.uniform(*r_constr)
    theta = random.uniform(*theta_constr)
    phi = random.uniform(*phi_constr)
    x, y, z = polar2cart(r, theta, phi)
    obj.location[0] = point.location[0] + x
    obj.location[1] = point.location[1] + y
    obj.location[2] = point.location[2] + z
    
def random_rotation(obj, constraints):
    rot_constr_1, rot_constr_2, rot_constr_3 = constraints
    new_rot_1 = random.uniform(*rot_constr_1)
    new_rot_2 = random.uniform(*rot_constr_2)
    new_rot_3 = random.uniform(*rot_constr_3)
    obj.rotation_euler[0] = new_rot_1
    obj.rotation_euler[1] = new_rot_2
    obj.rotation_euler[2] = new_rot_3

def get_bin_info(bin):
    quat = bin.rotation_euler.to_quaternion()
    return [
        bin.location[0], 
        bin.location[1], 
        bin.location[2], 
        quat[0],
        quat[1],
        quat[2],
        quat[3],
        bin.dimensions.x,
        bin.dimensions.y,
        bin.dimensions.z,
    ]

def get_camera_info(camera):
    quat = camera.rotation_euler.to_quaternion()
    return [
        camera.location[0],
        camera.location[1],
        camera.location[2], 
        quat[0],
        quat[1],
        quat[2],
        quat[3],
    ]

def make_image(camera, output_dir, filename):
    bpy.data.scenes['Scene'].camera = camera
    bpy.context.scene.render.filepath = os.path.join(output_dir, filename)
    bpy.ops.render.render(write_still = True)


#bpy.context.scene.use_nodes = True
#tree = bpy.context.scene.node_tree

#for node in tree.nodes:
#    tree.nodes.remove(node)

## create input image node
#render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')
#render_layers_node.location = 0, 0

## create output node
#viewer_node = tree.nodes.new('CompositorNodeViewer')
#viewer_node.location = 400, 0

## link nodes
#links = tree.links
#link = links.new(render_layers_node.outputs[2], viewer_node.inputs[0])    

render_output_dir = r'/Users/pocoder/Desktop/diploma_project/render_results'

# x, y, z
# default 0z for bins: -0.2 -0.15 -0.07
default_z_up = [-0.2, -0.15, -0.07]
bin_location_constraints = [
    [(-6.1, -5.7), (7.6, 7.9), (0.205, 0.205)],
    [(0, 2), (14, 15), (0, 0)],
    [(-8.6, -8), (6.5, 10), (0, 0)],
    [(-3, -1), (-24.5, -25), (0, 0)],
]
# rot1, rot2, rot3
bin_rotation_cosntraints = [
    [(0, 0), (0, 0), (0, 2*math.pi - 0.001)],
    [(0, 0), (0, 0), (0, 2*math.pi - 0.001)],
    [(0, 0), (0, 0), (0, 2*math.pi - 0.001)],
    [(0, 0), (0, 0), (0, 2*math.pi - 0.001)],
]
# location between spheres with constraints on phi and theta
camera_loc_constraints = [
    [(1, 3), (math.pi/2+0.1, 3*math.pi/2-0.1), (0.1, math.pi/2-0.1)],
    [(1, 4), (math.pi-0.1, 2*math.pi+0.1), (0.05, math.pi/2-0.2)],
    [(1, 3), (-math.pi/2-0.1, math.pi/2+0.1), (0.1, math.pi/2-0.1)],
    [(1, 3), (-0.1, math.pi+0.1), (0.1, math.pi/2-0.1)],
]
column_names = [
    'scene_id', 'render_id', 
    'bin_loc_x', 'bin_loc_y', 'bin_loc_z', 
    'bin_rot_quat_1', 'bin_rot_quat_2', 'bin_rot_quat_3', 'bin_rot_quat_4'
    'bin_dim_x', 'bin_dim_y', 'bin_dim_z',
    'camera_loc_x', 'camera_loc_y', 'camera_loc_z', 
    'camera_rot_quat_1', 'camera_rot_quat_2', 'camera_rot_quat_3', 'camera_rot_quat_4',
]
# Total number of render images = len(scene_idcs)*len(bins)*n 
scene_idcs = [0, 1, 2, 3]
n = 1
bins = [0, 1, 2]
data = []
depth_maps = {}
for scene_idx in scene_idcs:
    for bin_id in bins:
        bin_location_const = copy(bin_location_constraints[scene_idx])
        bin_location_const[2] = [val+default_z_up[bin_id] for val in bin_location_const[2]]
        bin = get_bin(bin_id)
        camera = get_camera(0)
        for i in range(n):
            color_background_objects()
            random_color(bin)
            random_location(bin, bin_location_const)
            random_rotation(bin, bin_rotation_cosntraints[scene_idx])
            random_location_in_sphere(camera, bin, camera_loc_constraints[scene_idx])
            look_at(camera, bin)
            cr_1 = camera.rotation_euler[0]
            cr_2 = camera.rotation_euler[1]
            cr_3 = camera.rotation_euler[2]
            defl = 18/180 * math.pi
            random_rotation(camera, [
                (cr_1-defl, cr_1+defl),
                (cr_2-defl, cr_2+defl), 
                (cr_3-defl, cr_3+defl)
            ])
            data.append([scene_idx, n*bin_id+i, *get_bin_info(bin), *get_camera_info(camera)])
            make_image(camera, render_output_dir,
                       f'scene_{scene_idx}_render_{n*bin_id+i}.png')
#            pixels = bpy.data.images['File Output Node'].pixels
#            print(np.array(pixels[:]).shape)
#            arr = np.array(pixels[:]).reshape((1080, 1920, 4))[::-1,:,0]
#            depth_maps[f'scene_{scene_idx}_render_{n*bin_id+i}_depth'] = arr
        random_location(bin, [(-100, -100), (-100, -100), (-100, -100)])

# np.savez(os.path.join(render_output_dir, 'arr.npz'), **depth_maps)

with open(os.path.join(render_output_dir, 'data.csv'), 'w', newline='') as csvfile:
    datawriter = csv.writer(csvfile, delimiter=' ')
    datawriter.writerow(column_names)
    for row in data:
        datawriter.writerow(row)
