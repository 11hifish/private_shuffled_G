r"""
Three tasks: mean estimation, linear regression, logistic regression.
This file contains the objective function and the gradient function for each task.
"""

import numpy as np
from typing import Tuple, Callable, Optional


def mean_estimation_loss_fn(x: np.ndarray, true_dataset: np.ndarray) -> float:
    """
    Mean Estimation: F(x) = 0.5 sum_{i=1}^{n} |x - d_i|_2^2.

    Args:
        x: current model parameter to be evaluated.
        true_dataset: the dataset whose mean is to be estimated.

    Returns:
        F(x) - F(x^*), where x^* is the mean of true_dataset.
    """
    x_gt = np.mean(true_dataset, axis=0)
    return np.mean([(np.linalg.norm(x - d) ** 2) for d in true_dataset]) * 0.5 \
           - np.mean([(np.linalg.norm(x_gt - d) ** 2) for d in true_dataset]) * 0.5


def mean_estimation_gradient_fn(x: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """
    The gradient of mean estimation on a single sample: nabla f(x) = x - d

    Args:
        x: current model parameter.
        sample: a single sample on which the gradient is to be computed.

    Returns:
        The gradient vector.
    """
    assert (len(x) == len(sample))
    return x - sample


def linear_regression_loss_fn(x: np.ndarray, true_dataset: Tuple[np.ndarray, np.ndarray],
                              reg_unscaled: Optional[Callable[[np.ndarray], float]] = None,
                              scaler: Optional[float] = None) -> float:
    """
    Linear regression: F(x) = |A x - y|^2 = sum_{i=1}^{n} (<A_i, x> - y_i)^2, where true_dataset = (A, y)
        and the objective is G(x) = F(x) + scaler * reg_unscaled(x), if reg_unscaled is not None.
        If reg_unscaled is not None and scaler is None, scaler = 1.0 by default.

    Args:
        x: current model parameter.
        true_dataset: a tuple of the feature matrix A and the response vector y.

    Returns:
        G(x)
    """
    (A, y) = true_dataset
    # x_gt = np.linalg.pinv(A.T @ A) @ A.T @ y
    reg_x, reg_x_gt = 0., 0.
    if reg_unscaled is not None:
        if scaler is None:
            scaler = 1.0
        reg_x = scaler * reg_unscaled(x)
        # reg_x_gt = scaler * reg_unscaled(x_gt)
    return np.mean((A @ x - y) ** 2) + reg_x
           # - (np.linalg.norm(A @ x_gt, ord=2) ** 2 + reg_x_gt)


def linear_regression_gradient_fn(x: np.ndarray, sample: Tuple[np.ndarray, float]) -> np.ndarray:
    """
    The gradient of linear regression on a single sample: nabla f(x) = 2 (<a, x> - y) * a
    where (a, y) = sample.

    Args:
        x: current model parameter.
        sample: a single sample, i.e., a tuple consisting of a feature vector a and its response y.

    Returns:
        The gradient vector.
    """
    (a, y) = sample
    return 2 * (np.dot(a, x) - y) * a


def sigmoid(z):
    return 1. / (1 + np.exp(-z))

def logistic_regression_loss_fn(x: np.ndarray, true_dataset: Tuple[np.ndarray, np.ndarray],
                                reg_unscaled: Optional[Callable[[np.ndarray], float]] = None,
                                scaler: Optional[float] = None
                                ) -> float:
    """
       Logistic regression: F(x) = sum_{i=1}^{n} y \log y_hat + (1-y) \log (1-y_hat),
           where y_hat = sigmoid(<x, a>) is the prediction for parameter x and feature vector a.
           The objective is G(x) = F(x) + scaler * reg_unscaled(x), if reg_unscaled is not None.
           If reg_unscaled is not None and scaler is None, scaler = 1.0 by default.

       Args:
           x: current model parameter.
           true_dataset: a tuple of the feature matrix A and the response vector y.

       Returns:
           G(x)
       """
    (A, y) = true_dataset
    pred = sigmoid(A @ x)
    loss = -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
    reg_val = 0.
    if reg_unscaled is not None:
        if scaler is None:
            scaler = 1.0
        reg_val = scaler * reg_unscaled(x)
    return loss + reg_val


def logistic_regression_gradient_fn(x: np.ndarray, sample: Tuple[np.ndarray, float]) -> np.ndarray:
    """
        The gradient of logistic regression on a single sample: nabla f(x) = (y_hat - y) a
        where (a, y) = sample and y_hat = sigmoid(<a, x>).

        Args:
            x: current model parameter.
            sample: a single sample, i.e., a tuple consisting of a feature vector a and its response y.

        Returns:
            The gradient vector.
        """
    (a, y) = sample
    y_hat = sigmoid(np.dot(x, a))
    return (y_hat - y) * a


tasks_fn_map = {
    'mean_estimation': (mean_estimation_loss_fn, mean_estimation_gradient_fn),
    'logistic_regression': (logistic_regression_loss_fn, logistic_regression_gradient_fn),
    'linear_regression': (linear_regression_loss_fn, linear_regression_gradient_fn)
}
