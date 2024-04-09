from src.detector import PackageDetector, BottleDetector
import numpy as np
import cv2
from PIL import Image
import os


def package_optimization(alpha, beta, gamma):
    '''
    The function that will be passed into the optimizer to find the proper
    alpha, beta, and gamma parameters for the package.

    :param alpha: The first parameter controlling the contrast of the image.
    Allowed values are alpha >= 0.
    Use 0 <= alpha < 1 to decrease the contrast of the image.
    Use alpha > 1 to increase the contrast of the image.
    
    :param beta: The second parameter controlling the brightness of the image.
    Allowed values are -127 <= beta <= 127.
    
    :param gamma: The third parameter controlling the gamma correction (exposure) of the image.
    Allowed values are gamma >= 1.

    :return: The ration of the bottles the model detected devided than the bottles it should've
    detected
    '''
    
    # Hard code the number of packages the given video contains
    total_packages = 90 # (a package is detected more than once)

    # Setting the directory that contain the images
    video_path = os.path.join("./data/cropped_video.mp4")

    # Setting the capture object
    cap = cv2.VideoCapture(video_path)

    # Setting the size of the frames 
    ret, frame = cap.read()
    width, height = frame.shape[:2]

    # Initialize the detector object
    package_detector = PackageDetector(
        input_shape=(width, height, 3),
        padding={
            'top': 30,
            'bottom': 460,
            'left': 100,
            'right': 600,
        },
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )

    detected = 0
    while (ret):
        package = package_detector.detect_package(frame, _preproc=True)

        if package is not None:
            detected += 1

        ret, frame = cap.read()

    cap.release()

    return detected/total_packages


def bottle_optimization(alpha: float, beta: float, gamma: float) -> float:
    '''
    The function that will be passed into the optimizer to find the proper
    alpha, beta, and gamma parameters for the bottles.

    :param alpha: The first parameter controlling the contrast of the image.
    Allowed values are alpha >= 0.
    Use 0 <= alpha < 1 to decrease the contrast of the image.
    Use alpha > 1 to increase the contrast of the image.
    
    :param beta: The second parameter controlling the brightness of the image.
    Allowed values are -127 <= beta <= 127.
    
    :param gamma: The third parameter controlling the gamma correction (exposure) of the image.
    Allowed values are gamma >= 1.

    :return: The ration of the bottles the model detected devided than the bottles it should've
    detected
    '''

    # The number of bottles expected at every box
    num_bottles = 12

    # Initialize the detector object
    bottle_detector = BottleDetector(
        num_bottles=num_bottles,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )

    # Setting the directory that contain the images
    folder_path = os.path.join("./data")
    
    # Iterating over all images
    non_image_path = 0
    bottles_detected = 0
    for img_path in os.listdir(folder_path):
        if not img_path.endswith("jpg"):
            non_image_path += 1
            continue
        
        # Covnerting the image to np array
        image_path = os.path.join(folder_path, img_path)
        image = Image.open(image_path)
        img = np.array(image)

        # Calculating the bottles in the image
        bottles = bottle_detector.detect_bottles(img, _preproc=True)
        bottles_detected += bottles

    # Finding the total bottles the model should have detected
    with open(os.path.join(folder_path, "bottles.txt")) as f:
        total_truth_bottles = sum([int(num_b) for num_b in f.readlines()])    

    return bottles_detected/total_truth_bottles
