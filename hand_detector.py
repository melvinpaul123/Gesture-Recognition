import cv2
import numpy as np
from sys import platform
import pyopenpose as op

# Initialize OpenPose
params = dict()
params["model_folder"] = "path/to/openpose/models"  # Replace with the actual path to the models folder
params["hand"] = True
params["hand_detector"] = 2  # Use the OpenPose hand detector
params["hand_net_resolution"] = "320x320"  # Set the desired input resolution

# Check if running on Windows or MacOS
if platform == "win32" or platform == "darwin":
    # Windows or MacOS specific configuration
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()
else:
    # Linux specific configuration
    op_wrapper = op.WrapperPython(op.ThreadManagerMode.Synchronous)
    op_wrapper.configure(params)
    op_wrapper.start()

# Function to detect and crop hands
def detect_and_crop_hands(image):
    # Detect poses
    datum = op.Datum()
    datum.cvInputData = image
    op_wrapper.emplaceAndPop([datum])

    # Get hands keypoints
    hands = datum.handKeypoints

    # If hands are detected
    if hands is not None and len(hands) > 0:
        # Get the first hand keypoints
        keypoints = hands[0]

        # Calculate the bounding box for the hand keypoints
        min_x = np.min(keypoints[:, 0])
        max_x = np.max(keypoints[:, 0])
        min_y = np.min(keypoints[:, 1])
        max_y = np.max(keypoints[:, 1])

        # Crop the hand region
        hand_image = image[min_y:max_y, min_x:max_x]
        return hand_image

    return None

# Load image
image = cv2.imread('path/to/image.jpg')  # Replace with the actual path to the image

# Detect and crop hands
cropped_hands = detect_and_crop_hands(image)

# Display the original image and cropped hands
cv2.imshow('Original Image', image)
cv2.imshow('Cropped Hands', cropped_hands)
cv2.waitKey(0)
cv2.destroyAllWindows()
