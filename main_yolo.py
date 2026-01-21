import mujoco
import mujoco.viewer
import numpy as np
import cv2
import random
import os
import shutil
import re
from submission_yolo import detect_objects

objects_folder = "assignment_1/assets/objects"
coco_class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

available_objects = [d for d in os.listdir(objects_folder) if os.path.isdir(os.path.join(objects_folder, d))]
coco_available_objects = [obj for obj in available_objects if obj in coco_class_names]

num_to_select = min(len(coco_available_objects), 10)
selected_object_names = random.sample(coco_available_objects, num_to_select)
selected_objects = [{"name": name, "path": os.path.join(objects_folder, name), "coco_class": name} for name in selected_object_names]

# --- 2. Build XML ---
def build_xml_with_objects(selected_objects):
    xml_parts = []
    xml_parts.append('<?xml version="1.0" ?>')
    xml_parts.append('<mujoco model="turret_ultra_fast_drop">')
    
    # PHYSICS BOOST: Gravity set to -60 and Timestep to 0.001 for high-speed stability
    xml_parts.append('  <option timestep="0.001" gravity="0 0 -60"/>')
    xml_parts.append('  <compiler meshdir="assignment_1/assets/objects"/>')
    xml_parts.append('  <visual><global offwidth="1280" offheight="960"/></visual>')
    
    xml_parts.append('  <asset>')
    xml_parts.append('    <texture type="skybox" builtin="gradient" rgb1="0.1 0.1 0.1" rgb2="0 0 0" width="512" height="512"/>')
    xml_parts.append('    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".2 .2 .3" rgb2=".3 .3 .4"/>')
    xml_parts.append('    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0"/>')
    
    for obj in selected_objects:
        obj_name = obj["name"]
        assets_file = os.path.join(obj["path"], "assets.xml")
        if os.path.exists(assets_file):
            with open(assets_file, 'r') as f:
                content = f.read()
                asset_content = re.search(r'<asset>(.*?)</asset>', content, re.DOTALL)
                if asset_content:
                    meshes = re.findall(r'<mesh\s+name="([^"]+)"\s+file="([^"]+)"', asset_content.group(1))
                    for mesh_name, mesh_file in meshes:
                        xml_parts.append(f'    <mesh name="{obj_name}_{mesh_name}" file="{obj_name}/{os.path.basename(mesh_file)}"/>')
    xml_parts.append('  </asset>')
    
    xml_parts.append('  <worldbody>')
    xml_parts.append('    <light pos="0 0 5" dir="0 0 -1" directional="true"/>')
    xml_parts.append('    <geom name="floor" type="plane" size="10 10 0.1" material="grid" friction="1 0.05 0.01" solimp="0.9 0.95 0.001" solref="0.02 1"/>')
    
    # Turret
    xml_parts.append('    <body name="turret_base" pos="0 0 0">')
    xml_parts.append('        <geom type="cylinder" size=".1 .05" rgba=".5 .5 .5 1"/>')
    xml_parts.append('        <body name="pan_link" pos="0 0 .1">')
    xml_parts.append('            <joint name="pan" type="hinge" axis="0 0 1" damping="0.5" limited="true" range="-1.5 1.5"/>')
    xml_parts.append('            <geom type="capsule" fromto="0 0 0 0 0 .2" size=".04" rgba=".8 .8 .8 1" mass="1"/>')
    xml_parts.append('            <body name="tilt_link" pos="0 0 .2">')
    xml_parts.append('                <joint name="tilt" type="hinge" axis="0 1 0" damping="0.5" limited="true" range="-0.8 0.2"/>')
    xml_parts.append('                <geom type="box" size=".05 .02 .02" rgba=".8 .8 .8 1" mass="0.5"/>')
    xml_parts.append('                <camera name="eye" pos="0.1 0 0" euler="90 285 0" fovy="45"/>')
    xml_parts.append('                <geom type="cylinder" fromto="0 0 0 0.1 0 0" size=".03" rgba="0.1 0.1 0.1 1" mass="0.2"/>')
    xml_parts.append('            </body>')
    xml_parts.append('        </body>')
    xml_parts.append('    </body>')
    
    for idx, obj in enumerate(selected_objects):
        obj_name = obj["name"]
        row, col = idx // 3, idx % 3
        x, y = 1.5 + row * 1.2, (col - 1) * 1.2
        z = 2.0 + random.uniform(0, 0.5) 
        
        xml_parts.append(f'    <body name="obj_{obj_name}" pos="{x} {y} {z}">')
        xml_parts.append('        <freejoint/>') 
        
        body_file = os.path.join(obj["path"], "body.xml")
        visual_mesh_loaded = False
        
        if os.path.exists(body_file):
            with open(body_file, 'r') as f:
                content = f.read()
                geom_match = re.search(r'<geom[^>]*mesh="([^"]+)"[^>]*rgba="([^"]+)"', content)
                if geom_match:
                    mesh_ref, rgba_val = geom_match.group(1), geom_match.group(2)
                    xml_parts.append(f'        <geom type="mesh" mesh="{obj_name}_{mesh_ref}" rgba="{rgba_val}" mass="1.0"/>')
                    visual_mesh_loaded = True
                else:
                    mesh_only_match = re.search(r'<geom[^>]*mesh="([^"]+)"', content)
                    if mesh_only_match:
                        mesh_ref = mesh_only_match.group(1)
                        rand_color = f"{random.random():.2f} {random.random():.2f} {random.random():.2f} 1.0"
                        xml_parts.append(f'        <geom type="mesh" mesh="{obj_name}_{mesh_ref}" rgba="{rand_color}" mass="1.0"/>')
                        visual_mesh_loaded = True

        if not visual_mesh_loaded:
            xml_parts.append('        <geom type="sphere" size="0.1" rgba="0.5 0.5 0.5 1" mass="1.0"/>')
        xml_parts.append('    </body>')
        
    xml_parts.append('  </worldbody>')
    xml_parts.append('</mujoco>')
    return '\n'.join(xml_parts)

# --- 3. Run ---
xml_string = build_xml_with_objects(selected_objects)
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=960, width=1280)

tilt_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tilt")
data.qpos[model.jnt_qposadr[tilt_joint_id]] = -0.3

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            renderer.update_scene(data, camera="eye")
            display_img, results = detect_objects(renderer, data)
            viewer.sync()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except Exception as e:
    print(f"Simulation stopped: {e}")

cv2.destroyAllWindows()