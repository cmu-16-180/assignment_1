import mujoco
import mujoco.viewer
import numpy as np
import cv2
import random
import os
from ultralytics import YOLO

# Define COCO-compatible objects mapping (objects that YOLO can actually detect)
objects_folder = "objects"
coco_object_mapping = {
    'airplane': 'airplane',
    'apple': 'apple',
    'banana': 'banana',
    'binoculars': 'handbag',
    'bowl': 'bowl',
    'camera': 'camera',
    'coffeemug': 'cup',
    'cup': 'cup',
    'knife': 'knife',
    'book': 'book',
    'waterbottle': 'bottle',
    'clock': 'clock',
    'vase': 'vase',
    'teddybear': 'teddy bear',
    'toothbrush': 'toothbrush',
    'wineglass': 'wine glass',
    'scissors': 'scissors',
    'backpack': 'backpack',
    'mug': 'cup',
    'sportsball': 'sports ball',
    'alarmclock': 'clock',
    'teapot': 'cup',
    'duck': 'bird',
    'fryingpan': 'bowl',
    'toruslarge': 'bowl',
    'gamecontroller': 'remote',
    'doorknob': 'clock',
    'human': 'person',
    'elephant': 'elephant',
    'piggybank': 'bottle',
    'pyramidlarge': 'bowl',
    'stapler': 'scissors',
    'table': 'dining table',
    'toothpaste': 'toothbrush',
    'train': 'train',
    'watch': 'clock',
    'phone': 'cell phone',
}

# Get available objects that are in COCO dataset
available_objects = [d for d in os.listdir(objects_folder) if os.path.isdir(os.path.join(objects_folder, d))]
coco_available_objects = [obj for obj in available_objects if obj in coco_object_mapping]

print(f"Found {len(available_objects)} total objects in '{objects_folder}' folder")
print(f"Found {len(coco_available_objects)} COCO-compatible objects:")
print(coco_available_objects[:10], "..." if len(coco_available_objects) > 10 else "")

# Randomly select 5 objects from COCO-compatible ones
if len(coco_available_objects) >= 5:
    selected_object_names = random.sample(coco_available_objects, 5)
else:
    selected_object_names = coco_available_objects
    print(f"Warning: Only {len(coco_available_objects)} COCO objects available, using all")

selected_objects = [{"name": name, "path": os.path.join(objects_folder, name), "coco_class": coco_object_mapping[name]} for name in selected_object_names]

print(f"\nRandomly selected 5 COCO-compatible objects:")
for obj in selected_objects:
    print(f"  {obj['name']} -> {obj['coco_class']}")

# Build XML by reading and merging assets and bodies from selected objects
def build_xml_with_objects(selected_objects):
    import re
    xml_parts = []
    
    # Start XML with turret
    xml_parts.append('<?xml version="1.0" ?>')
    xml_parts.append('<mujoco model="turret_with_objects">')
    xml_parts.append('  <option timestep="0.005" gravity="0 0 -9.81"/>')
    xml_parts.append('  <compiler meshdir="objects"/>')
    xml_parts.append('  <visual>')
    xml_parts.append('    <global offwidth="1280" offheight="960"/>')
    xml_parts.append('  </visual>')
    
    # Collect all assets
    xml_parts.append('  <asset>')
    xml_parts.append('    <texture type="skybox" builtin="gradient" rgb1="0.1 0.1 0.1" rgb2="0 0 0" width="512" height="512"/>')
    xml_parts.append('    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".2 .2 .3" rgb2=".3 .3 .4"/>')
    xml_parts.append('    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0"/>')
    xml_parts.append('    <material name="glow_red" rgba="1 0 0 1" emission="1"/>')
    
    # Load assets from each object
    for obj in selected_objects:
        obj_name = obj["name"]
        obj_path = obj["path"]
        assets_file = os.path.join(obj_path, "assets.xml")
        
        if os.path.exists(assets_file):
            try:
                with open(assets_file, 'r') as f:
                    content = f.read()
                    # Extract content between <asset> tags
                    asset_content = re.search(r'<asset>(.*?)</asset>', content, re.DOTALL)
                    if asset_content:
                        # Extract all mesh definitions and correct paths
                        mesh_pattern = r'<mesh\s+name="([^"]+)"\s+file="([^"]+)"'
                        meshes = re.findall(mesh_pattern, asset_content.group(1))
                        for mesh_name, mesh_file in meshes:
                            # Get just the filename
                            filename = os.path.basename(mesh_file)
                            # Create unique mesh name with object prefix
                            unique_name = f"{obj_name}_{mesh_name}"
                            corrected_path = f"{obj_name}/{filename}"
                            xml_parts.append(f'    <mesh name="{unique_name}" file="{corrected_path}"/>')
            except Exception as e:
                print(f"Warning: Could not parse assets for {obj_name}: {e}")
    
    xml_parts.append('  </asset>')
    
    # Worldbody with turret and objects
    xml_parts.append('  <worldbody>')
    xml_parts.append('    <light pos="0 0 5" dir="0 0 -1" directional="true"/>')
    xml_parts.append('    <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>')
    
    # Add turret
    xml_parts.append('    <!-- THE ROBOT TURRET -->')
    xml_parts.append('    <body name="turret_base" pos="0 0 0">')
    xml_parts.append('        <geom type="cylinder" size=".1 .05" rgba=".5 .5 .5 1"/>')
    xml_parts.append('        <body name="pan_link" pos="0 0 .1">')
    xml_parts.append('            <joint name="pan" type="hinge" axis="0 0 1" damping="0.5" limited="true" range="-1.5 1.5"/>')
    xml_parts.append('            <geom type="capsule" fromto="0 0 0 0 0 .2" size=".04" rgba=".8 .8 .8 1" mass="1"/>')
    xml_parts.append('            <body name="tilt_link" pos="0 0 .2">')
    xml_parts.append('                <joint name="tilt" type="hinge" axis="0 1 0" damping="0.5" limited="true" range="-0.5 0.5"/>')
    xml_parts.append('                <geom type="box" size=".05 .02 .02" rgba=".8 .8 .8 1" mass="0.5"/>')
    xml_parts.append('                <camera name="eye" pos="0.1 0 0" euler="90 270 0" fovy="45"/>')
    xml_parts.append('                <geom type="cylinder" fromto="0 0 0 0.1 0 0" size=".03" rgba="0.1 0.1 0.1 1" mass="0.2"/>')
    xml_parts.append('            </body>')
    xml_parts.append('        </body>')
    xml_parts.append('    </body>')
    
    # Add objects with their actual meshes - positioned in front of camera FOV
    positions = [
        (0.8, -0.3, 0.1),  # Left side, in front
        (0.9, 0.3, 0.1),   # Right side, in front
        (1.0, 0, 0.1),     # Center, in front
        (1.1, -0.2, 0.1),  # Left-center, in front
        (1.0, 0.2, 0.1),   # Right-center, in front
    ]
    
    for idx, obj in enumerate(selected_objects):
        obj_name = obj["name"]
        obj_path = obj["path"]
        pos = positions[idx] if idx < len(positions) else (2 + idx*0.5, 0, 0.2)
        
        xml_parts.append(f'    <!-- Object: {obj_name} -->')
        xml_parts.append(f'    <body name="obj_{obj_name}" pos="{pos[0]} {pos[1]} {pos[2]}">')
        
        # Try to load body.xml to get the visual mesh
        body_file = os.path.join(obj_path, "body.xml")
        visual_mesh_loaded = False
        
        if os.path.exists(body_file):
            try:
                with open(body_file, 'r') as f:
                    content = f.read()
                    # Look for the main visual mesh (first mesh reference that's not "contact")
                    visual_match = re.search(r'<geom[^>]*name="[^"]*visual[^"]*"[^>]*mesh="([^"]+)"', content)
                    if visual_match:
                        visual_mesh_name = visual_match.group(1)
                        corrected_mesh = f"{obj_name}_{visual_mesh_name}"
                        xml_parts.append(f'        <geom type="mesh" mesh="{corrected_mesh}"/>')
                        visual_mesh_loaded = True
                    
                    # Also load the main object mesh (e.g., mesh="duck")
                    if not visual_mesh_loaded:
                        main_mesh = re.search(r'<geom[^>]*mesh="([^"]+)"(?![^>]*contact)', content)
                        if main_mesh:
                            main_mesh_name = main_mesh.group(1)
                            if 'contact' not in main_mesh_name:
                                corrected_mesh = f"{obj_name}_{main_mesh_name}"
                                xml_parts.append(f'        <geom type="mesh" mesh="{corrected_mesh}"/>')
                                visual_mesh_loaded = True
                                print(f"  Loaded main mesh '{main_mesh_name}' for {obj_name}")
            except Exception as e:
                print(f"  Warning: Could not parse body for {obj_name}: {e}")
        
        if not visual_mesh_loaded:
            print(f"  No visual mesh found for {obj_name}, using fallback sphere")
            xml_parts.append(f'        <geom type="sphere" size="0.1" rgba="0.5 0.5 0.5 1"/>')
        
        xml_parts.append('    </body>')
    
    xml_parts.append('  </worldbody>')
    
    # Actuators
    xml_parts.append('  <actuator>')
    xml_parts.append('    <position joint="pan" kp="50"/>')
    xml_parts.append('    <position joint="tilt" kp="50"/>')
    xml_parts.append('  </actuator>')
    
    xml_parts.append('</mujoco>')
    
    return '\n'.join(xml_parts)

xml_string = build_xml_with_objects(selected_objects)

# Load the model
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# Set up renderer for YOLO detection with higher resolution
renderer = mujoco.Renderer(model, height=960, width=1280)

# Load pre-trained YOLO model (not training, just using it for detection)
print("Loading pre-trained YOLOv8 model...")
print("(Downloading model if not already present...)")
try:
    yolo_model = YOLO('yolov8n.pt')  # Automatically downloads if not present
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Make sure you have an internet connection to download the model.")
    exit(1)

# Create a viewer for visualization

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Mujoco viewer and YOLO detection window started!")
        print("Press ESC in Mujoco window or 'q' in YOLO window to exit")
        
        while viewer.is_running():
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Render from camera for YOLO detection
            renderer.update_scene(data, camera="eye")
            img = renderer.render()
            
            # Convert BGR to RGB if needed
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
            
            # Run YOLO detection (non-training, just inference)
            results = yolo_model(img_rgb, verbose=False)
            
            # Get annotated image with bounding boxes and labels
            annotated_img = results[0].plot()
            
            # Convert back to BGR for OpenCV display
            annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
            
            # Display the annotated image in a separate window (resized for better visibility)
            display_img = cv2.resize(annotated_img_bgr, (1280, 960))
            cv2.imshow('YOLO Detection - Camera View', display_img)
            
            # Update viewer
            viewer.sync()
            
            # Break loop if 'q' is pressed in detection window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
except Exception as e:
    print(f"Viewer initialization failed: {e}")
    print("Running simulation without viewer...")
    
    # Fallback: run simulation without viewer, just YOLO detection
    for step in range(1000):  # Run for 1000 steps
        mujoco.mj_step(model, data)
        
        # Render from camera
        renderer.update_scene(data, camera="eye")
        img = renderer.render()
        
        # Convert BGR to RGB if needed
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        
        # Run YOLO detection
        results = yolo_model(img_rgb, verbose=False)
        
        # Get annotated image with bounding boxes and labels
        annotated_img = results[0].plot()
        
        # Convert back to BGR for OpenCV display
        annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        
        # Display the annotated image
        display_img = cv2.resize(annotated_img_bgr, (1280, 960))
        cv2.imshow('YOLO Detection - Camera View', display_img)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("Simulation ended.")
