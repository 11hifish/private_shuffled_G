import numpy as np


def soft_threshold(x, threshold):
    """Applies the soft-thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def l2_projection(x_n, C):
    """
    Projects x_n onto the L2 ball with radius C.

    Args:
        x_n (numpy array): The input vector.
        C (float): The radius of the L2 ball.

    Returns:
        numpy array: The projected vector.
    """
    norm_x_n = np.linalg.norm(x_n, ord=2)
    if norm_x_n <= C:
        return x_n
    else:
        return C * x_n / norm_x_n


def proximal_gradient_step(x_n, learning_rate, reg_unscaled, params):
    """
    Computes the proximal gradient step for x_n.

    Args:
        x_n (numpy array): Current point.
        learning_rate (float): Learning rate.
        reg_unscaled (callable): Regularization function (unscaled). Regularization = lambda_ * reg_unscaled(x).
        params (dict, optional): Additional parameters for psi. Usually contains "scaler" to scale the reg func.

    Returns:
        numpy array: Updated point after proximal gradient step.
    """
    scaler = params.get('scaler', 1.0)
    # Example: Soft-thresholding for L1 regularization
    if reg_unscaled.__name__ == "l1_norm":  # reg(x) = scaler * (|x|_1)
        return soft_threshold(x_n, scaler * learning_rate)

    # Example: Closed-form solution for L2 regularization
    elif reg_unscaled.__name__ == "l2_norm":  # reg(x) = scaler * (|x|_2^2 / 2)
        return x_n / (1 + scaler * learning_rate)

    # No regularization
    elif reg_unscaled.__name__ == 'no_reg':
        return x_n

    elif reg_unscaled.__name__ == 'l2_ball_proj':
        C = params.get('C', 1.0)  # Radius of the L2 ball
        return l2_projection(x_n, C)


    # Numerical optimization for general psi
    else:
        from scipy.optimize import minimize

        def objective(x):  # reg(x) = lambda_ * psi(x)
            return scaler * reg_unscaled(x) + np.sum((x - x_n) ** 2) / (2 * learning_rate)

        res = minimize(objective, x_n, method='L-BFGS-B')
        return res.x


# Example regularization functions
def l1_norm(x):
    return np.sum(np.abs(x))


def l2_norm(x):
    return np.sum(x ** 2) / 2


def no_reg(x):
    return 0


def l2_ball_proj(x, C=None):
    if not C:
        C = 1.0
    return 0 if np.linalg.norm(x, ord=2) <= C else np.inf


if __name__ == '__main__':

    # Example usage
    # x_n = np.array([3.0, -1.5, 0.0, 2.0])
    x_n = np.array([1.0, 3.0, 1.0, -1.0])
    learning_rate = 0.1
    scaler = 5
    params = {}
    params['scaler'] = scaler

    # Proximal step for L1 regularization
    x_new_l1 = proximal_gradient_step(x_n, learning_rate=learning_rate, reg_unscaled=l1_norm, params=params)
    print("Updated x with L1 regularization:", x_new_l1)

    # Proximal step for L2 regularization
    x_new_l2 = proximal_gradient_step(x_n, learning_rate=learning_rate, reg_unscaled=l2_norm, params=params)
    print("Updated x with L2 regularization:", x_new_l2)

    # Proximal step for L2 projection onto a ball
    params_proj = {}
    params_proj['scaler'] = 10
    params_proj['C'] = 3.0
    x_new_l2_proj = proximal_gradient_step(x_n, learning_rate=learning_rate, reg_unscaled=l2_ball_proj, params=params_proj)
    print("Updated x with L2 projection:", x_new_l2_proj)

