'''
A module containing the Detector class.
'''

import numpy as np
import cv2
import logging

from src.utils import is_package, modify_contrast, modify_exposure

from typing import Dict, Tuple, Union

Padding: Dict[str, int] = {
    'top': 0,
    'bottom': 0,
    'left': 0,
    'right': 0,
}


class Detector:
    '''
    A class that represents a detector.
    Uses custom cv algorithms to detect bottles.
    '''
    
    def __init__(self,
            # dataset parameters
            padding: Padding,
            input_shape: Tuple[int, int, int],
            num_bottles: int,
            
            # image preprocessing parameters
            alpha: float,
            beta: float,
            gamma: float = 1,
            
            # debug flag
            DEBUG: bool = False
        ):
        '''
        Initializes the Detector object.
        
        :param threshold_lines: The padding that will be applied to the threshold box.
        i.e. the top threshold line will be at the top of the image - top padding.
        
        :param input_shape: The shape of the input image.
        
        :param num_bottles: The number of bottles to detect.
        
        :param alpha: The first parameter controlling the contrast of the image.
        Allowed values are alpha >= 0.
        Use 0 <= alpha < 1 to decrease the contrast of the image.
        Use alpha > 1 to increase the contrast of the image.
        
        :param beta: The second parameter controlling the brightness of the image.
        Allowed values are -127 <= beta <= 127.
        
        :param gamma: The third parameter controlling the gamma correction (exposure) of the image.
        Allowed values are gamma >= 1.
        '''
        
        assert padding['top'] >= 0, 'The top padding must be greater than or equal to 0.'
        assert padding['bottom'] >= 0, 'The bottom padding must be greater than or equal to 0.'
        assert padding['left'] >= 0, 'The left padding must be greater than or equal to 0.'
        assert padding['right'] >= 0, 'The right padding must be greater than or equal to 0.'
        assert num_bottles >= 0, 'The number of bottles must be greater than or equal to 0.'
        assert alpha >= 0, 'The alpha parameter must be greater than or equal to 0.'
        assert -127 <= beta <= 127, 'The beta parameter must be between -127 and 127.'
        assert gamma >= 1, 'The gamma parameter must be greater than or equal to 1.'
        
        self.padding = padding
        self.input_shape = input_shape
        self.num_bottles = num_bottles
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # set up threshold lines
        height, width = input_shape[:2]
        
        self.threshold_lines = [
            # horizontal lines
            [
                # x1, y1
                (padding['left'], 0),
                # x2, y2
                (padding['left'], height),
            ],
            [
                # x1, y1
                (padding['right'], 0),
                # x2, y2
                (padding['right'], height),
            ],
            # vertical lines
            [
                # x1, y1
                (0, padding['top']),
                # x2, y2
                (width, padding['top']),
            ],
            [
                # x1, y1
                (0, padding['bottom']),
                # x2, y2
                (width, padding['bottom']),
            ],
        ]
        
        # set up logging based on DEBUG flag
        if DEBUG:
            logging.basicConfig(level=logging.DEBUG)
            logging.debug('DEBUG mode is on.')
        else:
            logging.basicConfig(level=logging.INFO)
    
    def _preprocess(self, img: np.array) -> np.array:
        '''
        Preprocesses the image.
        Modifies the contrast, brightness and exposure of the image.
        
        :param img: The image to preprocess.
        
        :return: The preprocessed image.
        '''
        
        # modify the contrast and brightness
        out = modify_contrast(img, self.alpha, self.beta)
        out = modify_exposure(out, self.gamma)
        return out
    
    def detect_bottles(self, img: np.array, _preproc: bool = False) -> int:
        '''
        Finds the number of bottles in the image.
        
        :param img: The image to find the bottles in.
        :param _preproc: If True, the image will be preprocessed before finding the bottles.
        
        :return: The number of bottles in the image.
        '''
        
        if _preproc:
            img = self._preprocess(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            th, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = dst

        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        
        width, height = self.input_shape[:2]
        
        # remove contours that are too big (package contour)
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w >= .6 * width or h >= .6 * height:
                continue
            valid_contours.append(contour)
        
        if len(contours) == 0:
            return 0

        # sort contous by area
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        # keep only the first num_bottles contours
        valid_contours = valid_contours[:self.num_bottles]

        # remove contours that are too small
        area_avg = cv2.contourArea(valid_contours[0])
        for i, contour in enumerate(valid_contours):
            area = cv2.contourArea(contour)
            if area < .5 * area_avg or area > 1.5 * area_avg:
                valid_contours.pop(i)
            else:
                area_avg = (area_avg + area) / 2

        return len(valid_contours)
    
    def detect_package(self, img: np.array, _preproc: bool = False, _return_modified: bool = True) -> Union[np.array, None]:
        '''
        Finds the package in the image.
        If no package is found, returns None.
        Otherwise, returns the cropped image of the package.
        
        :param img: The image to find the package in.
        :param _preproc: If True, the image will be preprocessed before finding the package.
        
        :return: The cropped (modified) image of the package or None. 
        '''
        if _preproc:
            img = self._preprocess(img)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        dv = self.padding['bottom'] - self.padding['top']
        dh = self.padding['right'] - self.padding['left']
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self._in_threshold_lines(x, y, w, h):
                
                if (w < 10 or h < 10):
                    continue
                
                if is_package(h, w, dv, dh):
                    package = dst[y:y+h, x:x+w] if _return_modified else img[y:y+h, x:x+w]
                    return package

        return None
    
    def _in_threshold_lines(self, x: int, y: int, w: int, h: int) -> bool:
        '''
        Returns True if the given bounding box is within the threshold lines.
        
        :param x: The x coordinate of the bounding box.
        :param y: The y coordinate of the bounding box.
        :param w: The width of the bounding box.
        :param h: The height of the bounding box.
        
        :return: True if the given bounding box is within the threshold lines.
        '''
        if x > self.padding['left'] and x + w < self.padding['right'] and y > self.padding['top'] and y + h < self.padding['bottom']:
            return True
        
        return False