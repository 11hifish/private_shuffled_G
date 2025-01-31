r"""
Mean estimation experiments.
Regularization psi(x) = l2 projection to a ball of radius C. In the experiments, C = 10.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
from data_util import mean_estimation_datasets
from tasks import mean_estimation_loss_fn, mean_estimation_gradient_fn
from geneneralized_shuffle_grad import DatasetPair, AlgoSpec, GeneralizedShuffledGradient
from prox_op import l2_ball_proj
from privacy_util import compute_privacy_loss_across_epochs
import pickle


# data name: {'synthetic', 'mnist69'}
#     - 'synthetic' / 'synthetic2': estimate mean of Gaussian vectors. True: N(0, 1), Public: N(0.01) or N(0.01)
#     - 'mnist69': estimate mean pixel intensity of a given digit. True: digit 6, Public: 180 rotated pixel 9,
#                or other digits.
# data_name = 'synthetic'
data_name = 'mnist69'
# data_name = 'synthetic-u'
# data_name = 'pacs'

res_folder = f'res_mean_est_{data_name}'
if not os.path.isdir(res_folder):
    os.mkdir(res_folder)

# Prepare / load datasets
if data_name == 'synthetic':
    true_dataset = mean_estimation_datasets(dataname='G0')
    public_dataset = mean_estimation_datasets(dataname='G001')
elif data_name == 'synthetic2':
    true_dataset = mean_estimation_datasets(dataname='A0')
    public_dataset = mean_estimation_datasets(dataname='A001')
elif data_name == 'synthetic-u':
    true_dataset = mean_estimation_datasets(dataname='U0')
    public_dataset = mean_estimation_datasets(dataname='U1')
elif data_name == 'mnist69':
    target_n_samples = 1000
    true_dataset = mean_estimation_datasets(dataname='mnist_6', dataset_params={'target_n_samples': target_n_samples})
    public_dataset = mean_estimation_datasets(dataname='mnist_9', dataset_params={'target_n_samples': target_n_samples, 'rot': True})
elif data_name == 'pacs':
    true_dataset, public_dataset = mean_estimation_datasets(dataname='pacs')
else:
    raise Exception(f'Unknown data name {data_name}!')

# Global (fixed) HPs
# num_epochs = 50
# eps = 5  # privacy loss
delta = 1e-6   # privacy failure prob.
x0 = np.zeros(true_dataset.shape[1])  # initialization
GC_norm = 10  # gradient clipping norm
C = 10  # proj grad norm
num_exps = 10  # num of repeats
# method_name = 'IG'  # IG, SO, RR
p = 0.75  # fraction of gradient steps using private samples

for eps in [10]:
    # for num_epochs in [50, 80, 100]:
    for num_epochs in [50]:
        for method_name in ['RR']:
            # HPs (grid search)
            learning_rate_candidates = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001,
                                        5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9]

            list_opt_algo_names = ['public_shuffled_g', 'dp_shuffled_g', 'priv_pub', 'pub_priv', 'interleaved']
            # list_opt_algo_names = ['plain', 'public_shuffled_g']

            # Results
            global_results = {}

            def run_one_exp(algo_spec: AlgoSpec, learning_rate: float, algo_GC_norm: Optional[float]):
                opt_algo = GeneralizedShuffledGradient(algo_spec=algo_spec, x0=x0, learning_rate=learning_rate,
                                                       loss_fn=mean_estimation_loss_fn,
                                                       gradient_fn=mean_estimation_gradient_fn,
                                                       method_name=method_name,  # does not matter when there is no private samples
                                                       unscaled_reg_fn=l2_ball_proj,
                                                       params_reg_fn={'C': C},
                                                       GC_norm=algo_GC_norm)
                return opt_algo.train()


            # Start experiments by first picking an optimization algo
            for opt_algo_name in list_opt_algo_names:
                # Construct data spec and algo spec based on opt algo
                if opt_algo_name == 'plain':
                    dsp = DatasetPair(private_dataset=true_dataset, public_dataset=None, n_d=None)
                    dataset_spec = [dsp for _ in range(num_epochs)]
                    noise_sigma = np.zeros(num_epochs)
                    algo_GC_norm = None  # using public dataset only, no need to apply grad clipping
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
                    n_d = int(p * len(true_dataset))  # number of private samples to use per epoch
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
                for learning_rate in learning_rate_candidates:
                    loss_this_lr = np.zeros((num_exps, num_epochs + 1))
                    for exp_idx in range(num_exps):
                        print(f"opt algo {opt_algo_name}, lr {learning_rate}, exp {exp_idx}")
                        x_out, recorder = run_one_exp(algo_spec=algo_spec, learning_rate=learning_rate, algo_GC_norm=algo_GC_norm)
                        loss_this_lr[exp_idx] = np.array(recorder['loss'])
                        init_loss = mean_estimation_loss_fn(x0, true_dataset)
                        output_loss = mean_estimation_loss_fn(x_out, true_dataset)
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
            res_path = f'mean_est_{data_name}_K_{num_epochs}_eps_{eps}_nexp_{num_exps}_m_{method_name}_p_{p}_gc_{GC_norm}.pkl'
            with open(os.path.join(res_folder, res_path), 'wb') as f:
                pickle.dump(global_results, f)
            f.close()


# Plot results
color_map = {
    'public_shuffled_g': 'g',
    'dp_shuffled_g': 'b',
    'priv_pub': 'orange',
    'pub_priv': 'violet',
    'interleaved': 'r',
    'plain': 'black'
}

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
# plt.ylabel("$G(x)$ - $G(x^*)$", fontsize=25)
# plt.show()
# plt.savefig('mean_est_synthetic_test.pdf', bbox_inches='tight', pad_inches=0.1)
# plt.close()
