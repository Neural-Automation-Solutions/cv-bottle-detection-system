from bayes_opt import BayesianOptimization
from numpy.random import RandomState

from typing import Callable, Dict, Tuple, Optional, Union


class Optimizer:
    def __init__(
        self,
        function: Callable,
        pbounds: Dict[str, Tuple[int]],
        random_state: Optional[Union[int, RandomState, None]]=69,
        verbose: bool=0
        ) -> None:
        """
        Optimizer is a wrapper around the BayesianOptimization class.

        :param function: Function to optimize
        :param pbounds: Bounding boxes of the parameters, e.g., {'x': (2, 4), 'y': (-3, 3)}
        :param random_state: Seed for the parameters
        :param verbose: Verbose level at maximazation
        """
        self.optimizer = BayesianOptimization(
            f=function,
            pbounds=pbounds,
            verbose=verbose,
            random_state=random_state
        )

    def maximize(self, n_iter:int, init_points: int) -> Dict[str, float]:
        """
        Maximizing the function by changing the values of the parameters

        :param n_iter: steps of bayesian optimization
        :param init_points: Steps of random exploration

        :return: Dictionary containing the function values for the
        parameters of each iteration
        """
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)

        return self.optimizer.res
    
    def get_max(self) -> Dict[str, float]:
        """
        Return the parameters that optimized the function

        :return: The optimal parameters
        """
        return self.optimizer.max['params']
