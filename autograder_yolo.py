import cv2
import os
import numpy as np
import submission_yolo

class MockRenderer:
    """Fakes the Mujoco Renderer to return static test images."""
    def __init__(self, image_path):
        self.image_path = image_path

    def update_scene(self, data, camera):
        # Do nothing, as there is no real Mujoco scene
        pass

    def render(self):
        # Load the specific test image requested for this test case
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Could not find test image at {self.image_path}")
        return img

def run_tests():
    test_folder = "assignment_1/assets/test_yolo"
    
    # Updated test cases: only using the 6 new images
    test_cases = [
        ("objects_detected_1_1.png", 1),
        ("objects_detected_1_2.png", 1),
        ("objects_detected_2_1.png", 2),
        ("objects_detected_2_2.png", 2),
        ("objects_detected_3_1.png", 3),
        ("objects_detected_3_2.png", 3)
    ]

    total_passed = 0
    total_tests = len(test_cases)

    print("--- Starting Gradescope-style YOLO Autograder ---")
    
    # Pre-load the model once
    submission_yolo.load_yolo_model()

    for filename, min_expected in test_cases:
        image_path = os.path.join(test_folder, filename)
        
        if not os.path.exists(image_path):
            print(f"⚠️ SKIPPING: {filename} not found in {test_folder}")
            continue

        print(f"\nTesting {filename} (Expecting at least {min_expected} detections)...")
        
        try:
            # Create the mock renderer with the specific image
            mock_renderer = MockRenderer(image_path)
            
            # Call the user's function
            _, results = submission_yolo.detect_objects(mock_renderer, None)
            
            # Get detection count from YOLO results
            detections = len(results[0].boxes)
            
            if detections >= min_expected:
                print(f"✅ PASS: Found {detections} objects.")
                total_passed += 1
            else:
                print(f"❌ FAIL: Found only {detections} objects, but expected at least {min_expected}.")
        
        except Exception as e:
            print(f"⚠️ ERROR during test for {filename}: {e}")

    # Final result reporting
    print("\n---------------------------------------------------------")
    if total_passed == total_tests and total_tests > 0:
        print("All test passed successfully and YOLO detection implementation is correct!")
    elif total_tests == 0:
        print("Error: No test cases found.")
    else:
        print(f"Test complete. {total_passed}/{total_tests} cases passed.")
    print("---------------------------------------------------------")
    
    return total_passed == total_tests

if __name__ == "__main__":
    run_tests()