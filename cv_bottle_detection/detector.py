'''
A module containing the Detector class.
'''

import numpy as np
import cv2
import logging

from cv_bottle_detection.utils import is_package, modify_contrast, modify_exposure, distance_segmentation

from typing import Dict, Tuple, Union, List


class BaseDetector:
    '''
    A base class that will be used to create the detectors.
    '''

    def __init__(self,
            # image preprocessing parameters
            alpha: float,
            beta: float,
            gamma: float = 1,
            
            # debug flag
            DEBUG: bool = False
        ):
        '''
        Initializes the Base Detector.
        
        :param alpha: The first parameter controlling the contrast of the image.
        Allowed values are alpha >= 0.
        Use 0 <= alpha < 1 to decrease the contrast of the image.
        Use alpha > 1 to increase the contrast of the image.
        
        :param beta: The second parameter controlling the brightness of the image.
        Allowed values are -300 <= beta <= 300.
        
        :param gamma: The third parameter controlling the gamma correction (exposure) of the image.
        Allowed values are gamma >= 1.
        '''

        assert alpha >= 0, 'The alpha parameter must be greater than or equal to 0.'
        assert -300 <= beta <= 300, 'The beta parameter must be between -300 and 300.'
        assert gamma >= 1, 'The gamma parameter must be greater than or equal to 1.'

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
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


class PackageDetector(BaseDetector):
    '''
    A class that represents a detector.
    Uses custom cv algorithms to detect packages.
    '''

    def __init__(self,
            # dataset parameters
            input_shape: Tuple[int, int, int],
            
            # image preprocessing parameters
            alpha: float,
            beta: float,
            delta: int,
            epsilon: int,
            gamma: float = 1,


            # initialize image padding
            padding: Dict[str, int] = {
                'top': 0,
                'bottom': 0,
                'left': 0,
                'right': 0,
            },
            
            # debug flag
            DEBUG: bool = False
        ):
        '''
        Initializes the Package Detector object.

        :param input_shape: The shape of the input image.
        
        :param threshold_lines: The padding that will be applied to the threshold box.
        i.e. the top threshold line will be at the top of the image - top padding.
        
        :param alpha: The first parameter controlling the contrast of the image.
        Allowed values are alpha >= 0.
        Use 0 <= alpha < 1 to decrease the contrast of the image.
        Use alpha > 1 to increase the contrast of the image.
        
        :param beta: The second parameter controlling the brightness of the image.
        Allowed values are -127 <= beta <= 127.
        
        :param gamma: The third parameter controlling the gamma correction (exposure) of the image.
        Allowed values are gamma >= 1.
        
        :param delta: The number of iterations for the erosion operation.
        Allowed values are 0 <= delta <= 20.
        
        :param epsilon: The threshold value for the thresholding operation.
        Allowed values are 0 <= epsilon <= 255.
        '''

        super().__init__(alpha, beta, gamma, DEBUG)

        assert 0 <= delta <= 20, 'The delta parameter must be between 0 and 20.'
        assert 0 <= epsilon <= 255, 'The epsilon parameter must be between 0 and 255.'

        self.delta = delta
        self.epsilon = epsilon

        assert padding['top'] >= 0, 'The top padding must be greater than or equal to 0.'
        assert padding['bottom'] >= 0, 'The bottom padding must be greater than or equal to 0.'
        assert padding['left'] >= 0, 'The left padding must be greater than or equal to 0.'
        assert padding['right'] >= 0, 'The right padding must be greater than or equal to 0.'

        self.padding = padding
        
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

    def detect_package(self, img: np.array, _preproc: bool = True, _return_modified: bool = False, _return_bbox=False) -> Union[np.ndarray, None, Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        '''
        Finds the package in the image.
        If no package is found, returns None.
        Otherwise, returns the cropped image of the package.
        
        :param img: The image to find the package in.
        :param _preproc: If True, the image will be preprocessed before finding the package.
        :param _return_modified: If True, the modified image will be returned.
        :param _return_bbox: If True, the bounding box of the package will be returned.
        
        :return: The cropped (modified) image of the package or None. 
        '''
        if _preproc:
            new_img = self._preprocess(img)
        
            # erode
            kernel = np.ones((5,5),np.uint8)
            new_img = cv2.erode(new_img,kernel,iterations = self.delta)
      
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
      
        th, dst = cv2.threshold(gray, self.epsilon, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

                    if _return_bbox:
                        return package, (x, y, w, h)
                    
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


class BottleDetector(BaseDetector):
    '''
    A class that represents a detector.
    Uses custom cv algorithms to detect bottles in a package.
    '''

    def __init__(self,
            # dataset parameters
            num_bottles: int,
            
            # image preprocessing parameters
            alpha: float,
            beta: float,
            gamma: float = 1,
            delta: int = 0,
            epsilon: int = 0,
            stigma: float = .1,
            
            # debug flag
            DEBUG: bool = False
        ):
        '''
        Initializes the Base Detector.
        
        :param num_bottles: The number of bottles to detect.
        
        :param alpha: The first parameter controlling the contrast of the image.
        Allowed values are alpha >= 0.
        Use 0 <= alpha < 1 to decrease the contrast of the image.
        Use alpha > 1 to increase the contrast of the image.
        
        :param beta: The second parameter controlling the brightness of the image.
        Allowed values are -127 <= beta <= 127.
        
        :param gamma: The third parameter controlling the gamma correction (exposure) of the image.
        Allowed values are gamma >= 1.
        
        :param delta: The number of iterations for the erosion operation.
        Allowed values are 0 <= delta <= 20.
        
        :param epsilon: The number of iterations for the opening operation.
        Allowed values are 0 <= epsilon <= 20.
        
        :param stigma: The lower % threshold value for the bounding box size.
        Allowed values are 0 <= stigma <= 1.
        '''
    
        super().__init__(alpha, beta, gamma, DEBUG)

        assert 0 <= delta <= 20, 'The delta parameter must be between 0 and 20.'
        assert 0 <= epsilon <= 20, 'The epsilon parameter must be between 0 and 20.'
        assert 0 <= stigma <= 1, 'The stigma parameter must be between 0 and 1.'
        
        self.delta = delta
        self.epsilon = epsilon
        self.stigma = stigma
        
        assert num_bottles >= 0, 'The number of bottles must be greater than or equal to 0.'

        self.num_bottles = num_bottles

    def detect_bottles(self, img: np.array, _preproc: bool = True, _return_bboxes=False) -> Union[int, List[Tuple[int, int, int, int]]]:
        '''
        Finds the number of bottles in the image.
        
        :param img: The image to find the bottles in.
        :param _preproc: If True, the image will be preprocessed before finding the bottles.
        :param _return_bboxes: If True, the bounding boxes of the bottles will be returned.

        :return: The number of bottles in the image.
        '''
        
        if _preproc:
            img = self._preprocess(img)
            
            # erode
            kernel = np.ones((5, 5),np.uint8)
            img = cv2.erode(img,kernel,iterations=self.delta)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            th, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # opening
            kernel = np.ones((3, 3), np.uint8)
            dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel, iterations=self.epsilon)

            img = dst
        
        contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []   
        height, width = img.shape[:2]
        
        nodes_to_remove = []
        # remove contours that are too small
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            _next = hierarchy[0][i][0]
            previous = hierarchy[0][i][1]
            first_child = hierarchy[0][i][2]
            parent = hierarchy[0][i][3]
            
            # check for thresholds
            check = w >= .9 * width or h >= .9 * height or w <= self.stigma * width or h <= self.stigma * height
            
            # check if contour touches the edge
            check = check or x == 0 or y == 0 or x + w >= width or y + h >= height
            
            if check:
                
                # update previous
                if previous != -1:
                    hierarchy[0][previous][1] = _next
                
                # update next
                if _next != -1:
                    hierarchy[0][_next][1] = previous
                
                # update children
                hierarchy[0][first_child][3] = parent
                next_child = hierarchy[0][first_child][0]
                
                while next_child != -1:
                    hierarchy[0][next_child][3] = parent
                    next_child = hierarchy[0][next_child][0]
                
                nodes_to_remove.append(i)
                
                continue
            
            valid_contours.append(contour)

        # this is only for the optimizer
        if hierarchy is not None and hierarchy[0] is not None:
            new_hierarchy = [list(node) for i, node in enumerate(hierarchy[0]) if i not in nodes_to_remove]
        else:
            return 0
        
        if len(valid_contours) == 0:
            return 0

        # remove contours that are inside other contours
        contours_to_remove = []
        for i, contour in enumerate(valid_contours):
            parent = new_hierarchy[i][3]
            if parent == -1:
                continue
            
            contours_to_remove.append(i)
            
            # fill contour with white
            cv2.drawContours(img, [contour], -1, (255, 255, 255), -1)
            
        valid_contours = [contour for i, contour in enumerate(valid_contours) if i not in contours_to_remove]

        # keep only the first num_bottles contours
        valid_contours = valid_contours[:self.num_bottles]

        # fill contours with white
        for contour in valid_contours:
            cv2.drawContours(img, [contour], -1, (255, 255, 255), -1)

        final = len(valid_contours)

        if _return_bboxes:
            bboxes = [cv2.boundingRect(contour) for contour in valid_contours]
            return final, bboxes

        return final
