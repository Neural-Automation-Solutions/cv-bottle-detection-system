import os
import json
import cv2

from src.detector import PackageDetector, BottleDetector
from src.optimizer import Optimizer

from typing import Tuple

def package_optimization(
        data_dir: str,
        annotation_file: str,
        height: int = 480,
        width: int = 640,
        n_iter: int = 10,
        init_points: int = 5,
        verbose: int = 0,
    ) -> Tuple[float, float, float]:
    '''
    Optimizes a PackageDetector object to detect packages in images.
    
    :param data_dir: The directory containing the images to be used for optimization.
    :param annotation_file: The JSON file containing the annotations for the images in the data_dir.
    
    :return: The optimized alpha, beta, and gamma parameters for the PackageDetector object.
    '''

    # load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    def loss_function(alpha: float, beta: float, gamma: float) -> float:
        '''
        The function that will be passed into the optimizer to find the proper
        alpha, beta, and gamma parameters for the packages.
        
        :param alpha: The first parameter controlling the contrast of the image.
        Allowed values are alpha >= 0.
        Use 0 <= alpha < 1 to decrease the contrast of the image.
        Use alpha > 1 to increase the contrast of the image.
        
        :param beta: The second parameter controlling the brightness of the image.
        Allowed values are -300 <= beta <= 300.
        
        :param gamma: The third parameter controlling the gamma correction (exposure) of the image.
        Allowed values are gamma >= 1.
        
        :return: The ration of the packages the model detected devided than the packages it should've
        detected
        '''
        
        # Initialize the detector object
        package_detector = PackageDetector(
            input_shape=(width, height, 3),
            padding={
                'top': 110,
                'bottom': 410,
                'left': 140,
                'right': 530,
            },
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
        
        correct_detection = 0
        for annotation in annotations:
            # read image
            img = cv2.imread(os.path.join(data_dir, annotation['filename']))
            
            # detect package
            package = package_detector.detect_package(img, _preproc=True)
            
            # Condition to prevent the same package to be counted twice
            if package is not None and annotation['package']:
                correct_detection += 1
            elif package is None and not annotation['package']:
                correct_detection += 1
        
        return correct_detection/len(annotations)

    # Initialize the optimizer
    optimizer = Optimizer(
        function=loss_function,
        pbounds={
            'alpha': (0, 10),
            'beta': (-300, 300),
            'gamma': (1, 10)
        },
        verbose=verbose
    )

    # Run the optimizer
    optimizer.maximize(n_iter=n_iter, init_points=init_points)
    
    # get the optimal parameters
    optimal_params = optimizer.get_max()
    
    return optimal_params['alpha'], optimal_params['beta'], optimal_params['gamma']


def bottle_optimization(
        data_dir: str,
        annotation_file: str,
        num_bottles: int,
        n_iter: int = 10,
        init_points: int = 5,
        verbose: int = 0,
    ) -> Tuple[float, float, float]:
    '''
    Optimizes a BottleDetector object to count the number of bottles in a packages.
    
    :param data_dir: The directory containing the images to be used for optimization.
    :param annotation_file: The JSON file containing the annotations for the images in the data_dir.
    
    :return: The optimized alpha, beta, and gamma parameters for the BottleDetector object.
    '''

    # load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    def loss_function(alpha: float, beta: float, gamma: float) -> float:
        '''
        The function that will be passed into the optimizer to find the proper
        alpha, beta, and gamma parameters for the bottle detector.
        
        :param alpha: The first parameter controlling the contrast of the image.
        Allowed values are alpha >= 0.
        Use 0 <= alpha < 1 to decrease the contrast of the image.
        Use alpha > 1 to increase the contrast of the image.
        
        :param beta: The second parameter controlling the brightness of the image.
        Allowed values are -300 <= beta <= 300.
        
        :param gamma: The third parameter controlling the gamma correction (exposure) of the image.
        Allowed values are gamma >= 1.
        
        :return: The ration of the bottles the model detected devided than the bottles it should've
        detected
        '''

        # Initialize the detector object
        bottle_counter = BottleDetector(
            num_bottles=num_bottles,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

        acc = 0
        for annotation in annotations:
            # read image
            img = cv2.imread(os.path.join(data_dir, annotation['filename']))
            
            # count the number of bottles
            num_of_bottles = bottle_counter.detect_bottles(img, _preproc=True)

            acc += 1 - (abs(annotation['bottles'] - num_of_bottles) / annotation['bottles'])

        return acc/len(annotations)

    # Initialize the optimizer
    optimizer = Optimizer(
        function=loss_function,
        pbounds={
            'alpha': (0, 10),
            'beta': (-300, 300),
            'gamma': (1, 10)
        },
        verbose=verbose
    )

    # Run the optimizer
    optimizer.maximize(n_iter=n_iter, init_points=init_points)
    
    # get the optimal parameters
    optimal_params = optimizer.get_max()
    
    return optimal_params['alpha'], optimal_params['beta'], optimal_params['gamma']
