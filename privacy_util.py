import numpy as np
import matplotlib.pyplot as plt
import warnings


def _compute_per_epoch_privacy_loss_by_PABI(G, K, alpha, privacy_loss, amp_factor=1.0):
    # G:  Lipschitz parameter
    sigma_sq = 2 * alpha * (G ** 2) * K * amp_factor / privacy_loss
    return sigma_sq


def compute_privacy_loss_across_epochs(G, K, eps, delta, amp_factor=1.0, debug=False):
    # G: Lipschitz parameter, K: number of compositions / private epochs
    # (eps, delta) DP parameters
    # compose using RDP with HP alpha, then use grid search to get a good alpha that minimizes sigma
    alpha_lb = 1 + np.log(1/delta) / eps + 0.1
    alpha_ub = 60
    alpha_candidates = np.arange(alpha_lb, alpha_ub, 0.01)
    sigma_sq_candidates = []
    for alpha in alpha_candidates:
        privacy_loss = eps - np.log(1/delta) / (alpha - 1)
        ssc = _compute_per_epoch_privacy_loss_by_PABI(
            G=G, K=K, alpha=alpha,
            privacy_loss=privacy_loss,
            amp_factor=amp_factor)
        sigma_sq_candidates.append(ssc)
    min_idx = np.argmin(sigma_sq_candidates)
    min_sigma_sq = sigma_sq_candidates[min_idx]
    min_alpha = alpha_candidates[min_idx]
    if min_idx == len(alpha_candidates) - 1:
        warnings.warn("best alpha idx max, try increasing alpha candidate upper bound")
    if debug:
        assert ((np.array(sigma_sq_candidates) >= 0).all())
        print("number of alphas searched: ", len(sigma_sq_candidates))
        print("alpha lb: ", alpha_lb)
        print("min alpha: {}, min sigma sq: {}".format(min_alpha, min_sigma_sq))
        plt.plot(alpha_candidates, sigma_sq_candidates)
        plt.xlabel("alpha")
        plt.ylabel("sigma sq")
        plt.show()
    return min_sigma_sq, min_alpha


def main():
    G = 2
    K = 20
    eps = 5
    delta = 0.0001
    amp_factor = 1.0
    min_sigma_sq, min_alpha = \
        compute_privacy_loss_across_epochs(G=G, K=K, eps=eps, delta=delta, amp_factor=amp_factor, debug=True)


if __name__ == '__main__':
    main()
