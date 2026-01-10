# 16-180 - Concepts of Robotics - Assignment 1

Course: 16-180 Concepts of Robotics

Due Date: [Insert Date]

## 1. Overview

In this assignment, you will build the "eyes" for a robotic search-and-rescue turret. The robot is mechanically capable of moving (pan/tilt), but it is currently blind. Your job is to write a computer vision algorithm that locates a specific "Beacon" in a cluttered environment so the robot can lock onto it.

### Concepts Covered:
* Computer Vision (RGB vs. HSV color spaces).
* Thresholding and Binary Masks.
* Visual Servoing (controlling a robot based on camera feedback).

## 2. The Scenario
You are provided with a simulation of a Pan-Tilt Turret equipped with a camera.

* **The Target:** A Glowing Red Sphere (The Beacon).
* **The Distractors:** The environment contains "false positives" to trick your algorithm:
    * A Dark Red Cube (Correct Hue, Wrong Value/Shape).
    * An Orange Sphere (Wrong Hue, Right Shape).
* **The Goal:** Return the (x, y) pixel coordinates of the Beacon's center. If your code works, the instructor's control loop will automatically rotate the turret to keep the target centered.

## 3. Getting Started
### Step 1: Download the Files

In VS Code, open your `16-180_Concepts_of_Robotics` folder if it is not already open. Open a terminal and run the following:
```
git clone https://github.com/cmu-16-180/assignment_1.git
```
You should now see a new folder `assignment_1` with the following files:
* `main.py` (The Python code).
* `turret.xml` (The Simulation assets).

### Step 2: Run the Simulation
Open VS Code, activate your virtual environment, and run:

Windows/Linux:
```
python assignment/main.py
```

macOS:
```
mjpython assignment/main.py
```

### Step 3: What to Expect
* **Two Windows:** You should see two windows appear:
    1. **MuJoCo Viewer:** The main 3D simulation showing the robot and the room.
    2. **Robot Camera Feed (Debug Process):** A smaller window showing exactly what the robot sees.
* **Windows Users:** The first time you run this, you may get a **Windows Firewall** popup asking if Python can access the network.
    * **Action:** Click **"Allow Access".** (This is required because the two windows talk to each other using internal network sockets).
* **Initial Behavior:** The robot will likely stare at the center of the room or drift aimlessly. This is normal! You haven't written the vision code yet.

### 4. Your Task
Open assignment1.py in VS Code. You are only allowed to edit the function `find_target(image)`.
```
def find_target(image):
    # image is a (480, 640, 3) numpy array of RGB pixels
    ...
    return (cx, cy)
```
#### Implementation Strategy:
1. Color Space Conversion: The raw image is in RGB. It is often hard to detect "Red" in RGB because lighting changes the values. Converting to HSV (Hue, Saturation, Value) makes it much easier to isolate colors.
    * *Hint:* Use `cv2.cvtColor`.
2. Thresholding: Create a "mask" (a black and white image) where white pixels represent the target color and black pixels are everything else.
    * *Hint:* You might need two masks because "Red" in HSV wraps around the hue circle (both 0-10 and 170-180 are red).
3. Filtering: Eliminate the distractors.
    * The Dark Red Cube has the right color but is very dark (Low Value).
    * The Orange Sphere is bright but has the wrong Hue.
4. Centroid Calculation: Once you have a clean mask, calculate the center of the white blob.
    * *Hint:* Use `cv2.moments`.
    
### 5. Success Criteria

You know you are done when:

1. **Green Crosshair:** In the "Robot Camera Feed" window, a green circle stays locked onto the Red Sphere.
2. **Tracking:** The physical robot in the main window smoothly rotates to follow the sphere as it circles the room.
3. **Rejection:** The robot ignores the Red Cube and Orange Sphere, even when they cross the camera's view.

### 6. Interaction & Shutdown
* **To Quit:** Press **ESC** in the Camera Window or close the MuJoCo window. Both will shut down the program cleanly.
* **Troubleshooting:**
    * *The robot shakes violently:* Your vision code might be jumping between the target and a distractor. Check your thresholds.
    * *The Camera Window is black/frozen:* Check the terminal for errors. Ensure you aren't stuck in an infinite loop inside find_target.
    
### 7. Submission

Submit your modified assignment1.py file to Gradescope. (Do not submit the `turret.xml` or any other file).