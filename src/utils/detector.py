'''
A module containing utility functions for the detector.
'''

import cv2
import numpy as np

def modify_contrast(img: np.array, alpha: float, beta: float) -> np.array:
    '''
    Modifies the contrast and brightness of an image.
    Returns the modified image.
    
    :param img: The image to modify.
    :param alpha: The first parameter controlling the contrast of the image.
    Allowed values are alpha >= 0.
    Use 0 <= alpha < 1 to decrease the contrast of the image.
    Use alpha > 1 to increase the contrast of the image.
    
    :param beta: The second parameter controlling the brightness of the image.
    Allowed values are -127 <= beta <= 127.
    
    :return: The modified image.
    '''
    return cv2.addWeighted(img, alpha, img, 0, beta)
    # return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def modify_exposure(img: np.array, gamma: float) -> np.array:
    '''
    Modifies the exposure of an image.
    Returns the modified image.
    
    :param img: The image to modify.
    :param gamma: The parameter controlling the gamma correction (exposure) of the image.
    Allowed values are gamma >= 1.
    
    :return: The modified image.
    '''
    gamma_table=[np.power(x/255.0, gamma) * 255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def is_package(h: int, w: int, dv: int, dh: int, threshold = .7) -> bool:
    '''
    Returns True if the given bounding box are a package.
    
    :param h: The height of the bounding box.
    :param w: The width of the bounding box.
    :param dv: The vertical distance between the threshold lines.
    :param dh: The horizontal distance between the threshold lines.
    :param threshold: The threshold value for the distance.
    
    :return: True if the given bounding box are a package.
    '''
    
    if h < threshold * dv:
        return False
    if w < threshold * dh:
        return False
    
    return True

def distance_segmentation(img: np.array) -> int:
    '''
    Uses distance transform to segment the image.
    Assumes the image is binary.
    Returns the number of objects in the given image.
    
    For more on how this works, see:
    https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html
    
    :param img: The image to segment.
    
    :return: The number of objects in the given image.
    '''
    
    # Finding sure foreground area - area which we are sure is the object
    dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    # convert to uint8 type so that it can be used in findContours
    sure_fg = np.uint8(sure_fg)
    
    # get contours from sure_fg
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    area = sure_fg.shape[0] * sure_fg.shape[1]
    bbox_areas = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox_areas.append(h * w)
    contours = [c for index, c in enumerate(contours) if bbox_areas[index] > .1 * area]
    
    return len(contours)