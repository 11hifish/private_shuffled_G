import numpy as np
from prox_op import proximal_gradient_step
from typing import Union, List, Callable, Dict, Tuple
import copy

_DEBUG_MODE = False  # if true, print out intermediate steps

class DatasetPair(object):

    def __init__(self, private_dataset: Union[np.ndarray, Tuple[np.ndarray, np.ndarray], None] = None,
                 public_dataset: Union[np.ndarray, Tuple[np.ndarray, np.ndarray], None] = None,
                 n_d: Union[int, None] = None) -> None:
        self.private_dataset = private_dataset
        if private_dataset is None:
            self.n_private_samples = 0
        elif isinstance(private_dataset, tuple):
            self.n_private_samples = len(private_dataset[0])
        else:
            self.n_private_samples = len(private_dataset)
        self.public_dataset = public_dataset
        if public_dataset is None:
            self.n_public_samples = 0
        elif isinstance(public_dataset, tuple):
            self.n_public_samples = len(public_dataset[0])
        else:
            self.n_public_samples = len(public_dataset)
        self.num_samples = 0
        if n_d:
            assert (n_d <= self.n_private_samples)
        # number of private samples from the private dataset to be used.
        # If None, use all private samples in the private dataset
        self.n_d = n_d


def is_consistent_n_private_samples(dataset_spec: List[DatasetPair], target_num: int):
    if len(dataset_spec) == 0:
        return True
    if _DEBUG_MODE:
        print("target num in priv samples consistency check: ", target_num)
    for dspec in dataset_spec:
        if dspec.n_private_samples > 0 and dspec.n_private_samples != target_num:
            return False
    return True


class AlgoSpec(object):

    def __init__(self, dataset_spec: List[DatasetPair],
                 noise_sigma: np.ndarray,
                 true_dataset: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        :param dataset_spec: dataset used in each epoch
        :param noise_sigma: noise std. used in each epoch
        :param true_dataset: the dataset that defines the true objective
        """
        assert (len(dataset_spec) == len(noise_sigma))
        self.max_n_private_samples = max([despc.n_private_samples for despc in dataset_spec])
        # num of private samples must be the same across epochs if this num is not 0 (i.e., priv dataset is not None)
        assert (is_consistent_n_private_samples(dataset_spec, self.max_n_private_samples))
        self.dataset_spec = dataset_spec
        self.noise_sigma = noise_sigma
        self.num_epochs = len(dataset_spec)
        self.true_dataset = true_dataset


def generate_permutation(num_samples):
    return np.random.permutation(num_samples)


def clip_gradient(grad, GC_norm):
    grad_norm = np.linalg.norm(grad, ord=2)
    if grad_norm > GC_norm:
        return GC_norm * grad / grad_norm
    else:
        return grad


class GeneralizedShuffledGradient(object):

    def __init__(self, algo_spec: AlgoSpec,
                 x0: np.ndarray,
                 learning_rate: float,
                 loss_fn: Union[Callable[[np.ndarray, np.ndarray], float],
                                Callable[[np.ndarray, Tuple[np.ndarray, np.ndarray]], float]],
                 gradient_fn: Union[Callable[[np.ndarray, np.ndarray], np.ndarray],
                                    Callable[[np.ndarray, Tuple[np.ndarray, float]], np.ndarray]],
                 method_name: Union[str, None] = None,
                 unscaled_reg_fn: Union[Callable[[np.ndarray], float], None] = None,
                 params_reg_fn: Union[Dict, None] = None,
                 GC_norm: Union[float, None] = None):
        self.algo_spec = algo_spec  # dataset (aka. the objective) + noise per epoch
        self.true_dataset = self.algo_spec.true_dataset
        self.x0 = x0  # initialization
        self.learning_rate = learning_rate
        self.num_epochs = self.algo_spec.num_epochs
        self.loss_fn = loss_fn
        self.gradient_fn = gradient_fn
        self.method_name = method_name if method_name is not None else 'IG'  # use IG by default
        self.unscaled_reg_fn = unscaled_reg_fn
        self.params_reg_fn = params_reg_fn
        self.GC_norm = GC_norm

    def apply_proximal_step(self, current_x: np.ndarray, n_steps: int) -> np.ndarray:
        # apply regularization through the proximal step
        if not self.params_reg_fn:
            params = {}
            params['scaler'] = n_steps
        else:
            params = copy.deepcopy(self.params_reg_fn)
            if 'scaler' not in self.params_reg_fn:
                params['scaler'] = n_steps
            else:
                params['scaler'] = n_steps * params['scaler']
        new_x = proximal_gradient_step(x_n=current_x,
                                       learning_rate=self.learning_rate,
                                       reg_unscaled=self.unscaled_reg_fn,
                                       params=params)
        return new_x

    def train(self):
        n_private_samples = self.algo_spec.max_n_private_samples
        # initialize result recorder
        recorder = {}
        recorder['model_params'] = [self.x0]
        # compute initial loss
        init_loss = self.loss_fn(self.x0, self.true_dataset)
        recorder['loss'] = [init_loss]
        # default permutation
        private_samples_order = np.arange(n_private_samples)
        if self.method_name == 'SO':
            private_samples_order = generate_permutation(n_private_samples)
        x = self.x0
        if _DEBUG_MODE:
            print("private sample order (before training): ", private_samples_order)
        for k in range(self.num_epochs):
            # random reshuffling
            if self.method_name == 'RR':
                private_samples_order = generate_permutation(n_private_samples)
            if _DEBUG_MODE:
                print(f"private sample order in epoch {k}: ", private_samples_order)
            # get private and public dataset for this epoch
            private_ds_epoch = self.algo_spec.dataset_spec[k].private_dataset
            public_ds_epoch = self.algo_spec.dataset_spec[k].public_dataset
            if private_ds_epoch is not None and isinstance(private_ds_epoch, tuple):
                assert (len(private_ds_epoch[0] == len(private_ds_epoch[1])))
            if public_ds_epoch is not None and isinstance(public_ds_epoch, tuple):
                assert (len(public_ds_epoch[0] == len(public_ds_epoch[1])))
            # get noise for this epoch
            sigma = self.algo_spec.noise_sigma[k]
            if _DEBUG_MODE:
                print(f"noise sigma in epoch {k}: {sigma}")
            # begin training
            n_steps = 0
            # optimization using private samples
            if private_ds_epoch is not None:  # private dataset is not None
                n_priv_samples_to_use = self.algo_spec.dataset_spec[k].n_d
                if not n_priv_samples_to_use:  # n_d is None means using all samples from the private dataset
                    if isinstance(private_ds_epoch, tuple):
                        n_priv_samples_to_use = len(private_ds_epoch[0])
                    else:
                        n_priv_samples_to_use = len(private_ds_epoch)
                if _DEBUG_MODE:
                    print("n priv samples to use: ", n_priv_samples_to_use)
                for i in range(n_priv_samples_to_use):
                    # get private sample
                    priv_sample_idx = private_samples_order[i]
                    if isinstance(private_ds_epoch, tuple):
                        priv_sample = (private_ds_epoch[0][priv_sample_idx], private_ds_epoch[1][priv_sample_idx])
                    else:
                        priv_sample = private_ds_epoch[priv_sample_idx]
                    # compute gradient
                    grad = self.gradient_fn(x, priv_sample)
                    if self.GC_norm is not None:
                        grad = clip_gradient(grad=grad, GC_norm=self.GC_norm)
                    # sample noise
                    rho = np.random.normal(0, sigma, size=len(x))
                    # take one gradient step
                    x = x - self.learning_rate * (grad + rho)
                n_steps += n_priv_samples_to_use
            # optimization using public samples
            if public_ds_epoch is not None:  # public dataset is not None
                if isinstance(public_ds_epoch, tuple):
                    n_public_samples_to_use = len(public_ds_epoch[0])
                else:
                    n_public_samples_to_use = len(public_ds_epoch)
                if _DEBUG_MODE:
                    print("n public samples to use: ", n_public_samples_to_use)
                # always use the full public dataset without shuffling
                for j in range(n_public_samples_to_use):
                    if isinstance(public_ds_epoch, tuple):
                        pub_sample = (public_ds_epoch[0][j], public_ds_epoch[1][j])
                    else:
                        pub_sample = public_ds_epoch[j]
                    # compute gradient
                    grad = self.gradient_fn(x, pub_sample)
                    if self.GC_norm is not None:
                        grad = clip_gradient(grad=grad, GC_norm=self.GC_norm)
                    # sample noise
                    rho = np.random.normal(0, sigma, size=len(x))
                    # take one gradient step
                    x = x - self.learning_rate * (grad + rho)
                n_steps += n_public_samples_to_use
            # apply regularization
            if _DEBUG_MODE:
                print(f"n_steps in prox: {n_steps}")
            x = self.apply_proximal_step(current_x=x, n_steps=n_steps)
            # end of epoch: record model parameters and loss
            recorder['model_params'].append(x)
            current_loss = self.loss_fn(x, self.true_dataset)
            recorder['loss'].append(current_loss)
        return x, recorder

