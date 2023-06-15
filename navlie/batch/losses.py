"""Robust loss functions to be used in nonlinear least squares problems.

Robust loss functions here must inherit from the LossFunction interface, 
which defines a loss and a weight.
"""

from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    """
    Abstract base class for any loss function.
    """

    @abstractmethod
    def loss(self, e: float):
        """
        The loss function defines the cost :math:`\\rho(e)`, where :math:`e` is an
        error term and is often a function of the design variable.
        """
        pass

    @abstractmethod
    def weight(self, e: float):
        """
        The weight that depends on the current value of the error.
        This is used to reweight our original nonlinear least squares problem.
        """
        pass


class L2Loss(LossFunction):
    """
    Standard L2 loss. Simply 0.5 * e * e , with the robust
    weight of ones.
    """

    def loss(self, e: float):
        return 0.5 * e * e

    def weight(self, x: float):
        return np.ones(x.size)


class CauchyLoss(LossFunction):
    def __init__(self, c: float = 1.0):
        self.c = c

    def loss(self, e: float) -> float:
        """
        Cauchy loss function.

        The form here is taken from "MacTavish, Barfoot - At All Costs."

        Parameters
        ----------
        e : float
            Residual

        Returns
        -------
        float
            Cost evaluated at a given error.
        """
        return (0.5 * self.c**2) * np.log(1.0 + (e / self.c) ** 2)

    def weight(self, e: float) -> float:
        """
        Cauchy weight function.

        Parameters
        ----------
        e : float
            Residual

        Returns
        -------
        float
            Robust weight
        """
        return 1.0 / (1.0 + (e / self.c) ** 2)
