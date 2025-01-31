r"""
Linear regression experiments.
Regularization psi(x) = 0.5 * lambda_ * |x|_2^2 (l2 reg). lambda_ = 0.1.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple
from data_util import linear_regression_dataset
from tasks import linear_regression_loss_fn, linear_regression_gradient_fn
from geneneralized_shuffle_grad import DatasetPair, AlgoSpec, GeneralizedShuffledGradient
from prox_op import l2_norm, no_reg
from privacy_util import compute_privacy_loss_across_epochs
import pickle



# data name: {'blog_feedback'}
#     - 'blog_feedback': predict the number comments to be received. Pick n = 5000 samples.

# data_name = 'blog_feedback'
# data_name = 'wine'
# data_name = 'cali_housing'
data_name = 'crime_corrupted2'
# data_name = 'cifar10'


# data_name = 'synthetic'
res_folder = f'res_lin_reg_{data_name}'
if not os.path.isdir(res_folder):
    os.mkdir(res_folder)

# if data_name == 'synthetic':
#     X, y = make_regression(n_samples=5000, n_features=200, n_informative=100, noise=1, random_state=42)
#     true_dataset = (X, y)
#     public_dataset = None

if data_name == 'blog_feedback':
    (true_dataset, public_dataset) = linear_regression_dataset(
        dataname=data_name,
        dataset_params={'train_idx_path': 'blog_feedback_train_idx_5000.pkl',
                        'test_idx_path': 'blog_feedback_test_idx_5000.pkl'})
else:
    (true_dataset, public_dataset) = linear_regression_dataset(dataname=data_name)


# Global (fixed) HPs
num_epochs = 50
eps = 5  # privacy loss, eps = 1 or 5
delta = 1e-6   # privacy failure prob.
# x0 = np.zeros(true_dataset[0].shape[1])  # initialization
x0 = np.random.randn(true_dataset[0].shape[1]) * 0.01
GC_norm = 10  # gradient clipping norm
num_exps = 10  # num of repeats
method_name = 'RR'  # IG, SO, RR
p = 0.5  # fraction of gradient steps using private samples
reg_scaler = 0.1  # ridge regression, psi(x) = reg_scaler / 2 * |x|^2

# HPs (grid search)
learning_rate_candidates = [
    # 0.5, 0.1, 0.05,
    # 0.01, 0.005, 0.001,
    0.0005, 0.0001, 5e-5,
    1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8,
    # 5e-9, 1e-9
]
learning_rate_candidates_small = [
    # 0.01, 0.005, 0.001, 0.0005,
    # 0.0001, 5e-5,
    1e-5, 5e-6, 1e-6, 5e-7,
    1e-7, 5e-8, 1e-8, 5e-9, 1e-9]

list_opt_algo_names = ['public_shuffled_g', 'dp_shuffled_g', 'interleaved', 'priv_pub', 'pub_priv']
# list_opt_algo_names = ['public_shuffled_g', 'interleaved']

# Results
global_results = {}

# create the loss function w/ the target regularizer
def linear_regression_loss_fn_wrapper(x: np.ndarray, true_dataset: Tuple[np.ndarray, np.ndarray]) -> float:
    return linear_regression_loss_fn(x=x, true_dataset=true_dataset, reg_unscaled=l2_norm, scaler=reg_scaler)


def run_one_exp(algo_spec: AlgoSpec, learning_rate: float, algo_GC_norm: Optional[float]):
    opt_algo = GeneralizedShuffledGradient(algo_spec=algo_spec, x0=x0, learning_rate=learning_rate,
                                           loss_fn=linear_regression_loss_fn_wrapper,
                                           gradient_fn=linear_regression_gradient_fn,
                                           method_name=method_name,  # does not matter when there is no private samples
                                           unscaled_reg_fn=l2_norm,
                                           params_reg_fn={'scaler': reg_scaler},
                                           GC_norm=algo_GC_norm)
    return opt_algo.train()

# Start experiments by first picking an optimization algo
for opt_algo_name in list_opt_algo_names:
    # Construct data spec and algo spec based on opt algo
    if opt_algo_name == 'plain':
        dsp = DatasetPair(private_dataset=None, public_dataset=true_dataset, n_d=None)
        dataset_spec = [dsp for _ in range(num_epochs)]
        noise_sigma = np.zeros(num_epochs)
        algo_GC_norm = None
    elif opt_algo_name == 'public_shuffled_g':
        dsp = DatasetPair(private_dataset=None, public_dataset=public_dataset, n_d=None)
        dataset_spec = [dsp for _ in range(num_epochs)]
        noise_sigma = np.zeros(num_epochs)
        algo_GC_norm = None  # using public dataset only, no need to apply grad clipping
    elif opt_algo_name == 'dp_shuffled_g':
        dsp = DatasetPair(private_dataset=true_dataset, public_dataset=None, n_d=None)
        dataset_spec = [dsp for _ in range(num_epochs)]
        min_sigma_sq, min_alpha = compute_privacy_loss_across_epochs(G=2 * GC_norm,
                                                                     K=num_epochs, eps=eps, delta=delta, amp_factor=1.0)
        print(f"min sigma sq: {min_sigma_sq}, min alpha: {min_alpha}, eps: {eps}")
        noise_sigma = np.ones(num_epochs) * np.sqrt(min_sigma_sq)
        algo_GC_norm = GC_norm
    elif opt_algo_name == 'pub_priv':
        S = int((1 - p) * num_epochs)  # number of epochs on public dataset
        print(f"# epochs on public ds: {S}, # epochs on private ds: {num_epochs - S}")
        dsp_pub_only = DatasetPair(private_dataset=None, public_dataset=public_dataset, n_d=None)
        dsp_priv_only = DatasetPair(private_dataset=true_dataset, public_dataset=None, n_d=None)
        dataset_spec = [dsp_pub_only for _ in range(S)] + [dsp_priv_only for _ in range(num_epochs - S)]
        min_sigma_sq, min_alpha = compute_privacy_loss_across_epochs(G=2 * GC_norm,
                                                                     K=num_epochs - S, eps=eps, delta=delta, amp_factor=1.0)
        print(f"min sigma sq: {min_sigma_sq}, min alpha: {min_alpha}, eps: {eps}")
        noise_sigma = np.concatenate((np.zeros(S), np.ones(num_epochs - S) * np.sqrt(min_sigma_sq)))
        print("noise sigma: ", noise_sigma)
        algo_GC_norm = GC_norm
    elif opt_algo_name == 'priv_pub':
        S = int(p * num_epochs)  # number of epochs on private dataset
        print(f"# epochs on private ds: {S}, # epochs on public ds: {num_epochs - S}")
        dsp_priv_only = DatasetPair(private_dataset=true_dataset, public_dataset=None, n_d=None)
        dsp_pub_only = DatasetPair(private_dataset=None, public_dataset=public_dataset, n_d=None)
        dataset_spec = [dsp_priv_only for _ in range(S)] + [dsp_pub_only for _ in range(num_epochs - S)]
        min_sigma_sq, min_alpha = compute_privacy_loss_across_epochs(G=2 * GC_norm,
                                                                     K=S, eps=eps, delta=delta,
                                                                     amp_factor=1.0)
        print(f"min sigma sq: {min_sigma_sq}, min alpha: {min_alpha}, eps: {eps}")
        noise_sigma = np.concatenate((np.ones(S) * np.sqrt(min_sigma_sq), np.zeros(num_epochs - S)))
        print("noise sigma: ", noise_sigma)
        algo_GC_norm = GC_norm
    elif opt_algo_name == 'interleaved':
        if isinstance(true_dataset, tuple):
            n_private_samples = len(true_dataset[0])
        else:
            n_private_samples = len(true_dataset)
        n_d = int(p * n_private_samples)  # number of private samples to use per epoch
        if isinstance(public_dataset, tuple):
            n_public_samples = len(public_dataset[0])
        else:
            n_public_samples = len(public_dataset)
        n_public_samples_to_use = n_public_samples - n_d
        print(f"n priv samples per epoch : {n_d}, n public samples per epoch: {n_public_samples_to_use}")
        dsp_hybrid = DatasetPair(private_dataset=true_dataset,
                                 public_dataset=public_dataset[:n_public_samples_to_use], n_d=n_d)
        dataset_spec = [dsp_hybrid for _ in range(num_epochs)]
        min_sigma_sq, min_alpha = compute_privacy_loss_across_epochs(G=2 * GC_norm, K=num_epochs,
                                                                     eps=eps, delta=delta,
                                                                     amp_factor=1 / (n_public_samples_to_use + 1))
        print(f"min sigma sq: {min_sigma_sq}, min alpha: {min_alpha}, eps: {eps}")
        noise_sigma = np.ones(num_epochs) * np.sqrt(min_sigma_sq)
        algo_GC_norm = GC_norm
    else:
        raise Exception(f'Unknown opt algo name {opt_algo_name}!')
    # Create algo spec
    algo_spec = AlgoSpec(dataset_spec=dataset_spec, noise_sigma=noise_sigma, true_dataset=true_dataset)

    # HP search on learning rate
    best_lr = None
    best_loss_mean = None
    best_loss_std = None
    if opt_algo_name == 'public_shuffled_g' or opt_algo_name == 'plain':
        lr_candidates = learning_rate_candidates_small
        num_exps_algo = 1
    else:
        lr_candidates = learning_rate_candidates
        num_exps_algo = num_exps
    for learning_rate in lr_candidates:
        loss_this_lr = np.zeros((num_exps_algo, num_epochs + 1))
        for exp_idx in range(num_exps_algo):
            print(f"opt algo {opt_algo_name}, lr {learning_rate}, exp {exp_idx}")
            x_out, recorder = run_one_exp(algo_spec=algo_spec, learning_rate=learning_rate, algo_GC_norm=algo_GC_norm)
            loss_this_lr[exp_idx] = np.array(recorder['loss'])
            init_loss = linear_regression_loss_fn_wrapper(x0, true_dataset)
            output_loss = linear_regression_loss_fn_wrapper(x_out, true_dataset)
            print(f"init loss: {init_loss}, output loss: {output_loss}")
        print(f"loss this lr shape: {loss_this_lr.shape}")
        mean_loss_this_lr = np.mean(loss_this_lr, axis=0)
        std_loss_this_lr = np.std(loss_this_lr, axis=0)
        if (best_lr is None) or (mean_loss_this_lr[-1] < best_loss_mean[-1]):
            best_lr = learning_rate
            best_loss_mean = mean_loss_this_lr
            best_loss_std = std_loss_this_lr
    # add opt algo results to global results
    global_results[opt_algo_name] = (best_lr, best_loss_mean, best_loss_std)
# Save results
res_path = f'lin_reg_{data_name}_K_{num_epochs}_eps_{eps}_nexp_{num_exps}_m_{method_name}_p_{p}_gc_{GC_norm}.pkl'
with open(os.path.join(res_folder, res_path), 'wb') as f:
    pickle.dump(global_results, f)
f.close()

# Plot results
color_map = {
    'plain': 'black',
    'public_shuffled_g': 'g',
    'dp_shuffled_g': 'b',
    'priv_pub': 'orange',
    'pub_priv': 'violet',
    'interleaved': 'r'
}

# plt.figure(figsize=(6, 4))
# plt.figure(figsize=(8, 6))
# for opt_algo_name in list_opt_algo_names:
#     lr, loss_mean, loss_std = global_results[opt_algo_name]
#     plt.plot(np.arange(num_epochs + 1), loss_mean,
#              label=opt_algo_name + f', lr={lr}, method={method_name}',
#              c=color_map[opt_algo_name])
#     plt.fill_between(np.arange(num_epochs + 1), loss_mean - loss_std, loss_mean + loss_std, alpha=0.5, color=color_map[opt_algo_name])
# plt.legend(fontsize=12)
# plt.grid()
# plt.title(f'{data_name} ($\\epsilon$={eps})', fontsize=25)
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)
# plt.xlabel("# Epochs", fontsize=25)
# plt.ylabel("$G(x)$", fontsize=25)
# plt.show()


