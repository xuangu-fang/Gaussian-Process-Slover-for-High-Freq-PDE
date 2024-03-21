import numpy as np
import random
import matplotlib.pyplot as plt
import optax
import jax
import jax.numpy as jnp

import kernel_matrix
from kernel_matrix import *
from jax import vmap
import pandas as pd
import utils
import copy

import time
import tqdm
import os
import copy
import init_func

from model_GP_solver_1d import GP_solver_1d_single

import yaml
import fire
# from infras.misc import create_path
from infras.exp_config import ExpConfig

'''GP solver for 1d equation with a extra GP, which can accelerate the convergence by capturing the low frequency part of the solution quickly, designed for the hard cases of poission-1d:  sin(x) + 0.1*sin(20x) + 0.05*cos(100x), and sin(500x)-2*(x-0.5)^2'''


class GP_solver_1d_extra(GP_solver_1d_single):

    def __init__(self,
                 Xind,
                 y,
                 X_col,
                 src_col,
                 jitter,
                 X_test,
                 Y_test,
                 trick_paras=None,
                 fix_dict=None):
        super().__init__(Xind, y, X_col, src_col, jitter, X_test, Y_test,
                         trick_paras, fix_dict)

        self.cov_func_extra = trick_paras['kernel_extra']()
        self.kernel_matrix_extra = Kernel_matrix(self.jitter,
                                                 self.cov_func_extra)
        self.optimizer_extra = optax.adam(learning_rate=trick_paras['lr'])

        self.params = None
        self.params_extra = None

        print('using extra GP with kernel:',
              self.cov_func_extra.__class__.__name__)

    @partial(jit, static_argnums=(0, ))
    def value_and_grad_kernel_extra(self, params_extra, key):
        '''compute the value of the kernel matrix, along with Kinv_u and u_xx'''

        u = params_extra['u']  # function values at the collocation points
        kernel_paras = params_extra['kernel_paras']
        x_p = jnp.tile(self.X_con.flatten(), (self.N_con, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        # only the cov matrix of func vals
        K = self.kernel_matrix_extra.get_kernel_matrix(X1_p, X2_p,
                                                       kernel_paras)

        Kinv_u = jnp.linalg.solve(K, u)

        K_dxx1 = vmap(self.cov_func_extra.DD_x1_kappa,
                      (0, 0, None))(X1_p, X2_p, kernel_paras).reshape(
                          self.N_con, self.N_con)
        u_xx = jnp.matmul(K_dxx1, Kinv_u)

        return K, Kinv_u, u_xx

    @partial(jit, static_argnums=(0, ))
    def boundary_and_eq_gap_extra(self, u, u_extra, u_xx, u_xx_extra):
        """compute the boundary and equation gap, to construct the training loss or computing the early stopping criteria"""
        # boundary
        boundary_gap = jnp.sum(
            jnp.square(u[self.Xind].reshape(-1) +
                       u_extra[self.Xind].reshape(-1) - self.y.reshape(-1)))
        # equation
        if self.eq_type == 'poisson_1d':

            eq_gap = jnp.sum(
                jnp.square(u_xx.flatten() + u_xx_extra.flatten() -
                           self.src_col.flatten()))

        elif self.eq_type == 'allencahn_1d':

            u = u + u_extra
            eq_gap = jnp.sum(
                jnp.square(u_xx.flatten() + u_xx_extra.flatten() +
                           (u *
                            (u**2 - 1)).flatten() - self.src_col.flatten()))

        else:
            raise NotImplementedError

        return boundary_gap, eq_gap

    @partial(jit, static_argnums=(0, ))
    def loss_extra(self, params_extra, key):

        u = self.params['u']
        K, Kinv_u, u_xx = self.value_and_grad_kernel(self.params, key)

        u_extra = params_extra[
            'u']  # function values at the collocation points
        u_extra = u_extra.sum(axis=1).reshape(-1, 1)  # sum over trick
        log_v_extra = params_extra['log_v']  # inverse variance for eq ll
        log_tau_extra = params_extra[
            'log_tau']  # inverse variance for boundary ll
        kernel_paras_extra = params_extra['kernel_paras']

        K_extra, Kinv_u_extra, u_xx_extra = self.value_and_grad_kernel_extra(
            params_extra, key)

        boundary_gap, eq_gap = self.boundary_and_eq_gap_extra(
            u, u_extra, u_xx, u_xx_extra)

        log_prior = -0.5 * \
            jnp.linalg.slogdet(
                K_extra)[1]*self.trick_paras['logdet'] - 0.5*jnp.sum(u_extra*Kinv_u_extra)

        # boundary
        log_boundary_ll = 0.5 * self.N * log_tau_extra - 0.5 * \
            jnp.exp(
                log_tau_extra) * boundary_gap  # log likelihood of boundary

        # equation

        eq_ll = 0.5 * self.N_con * log_v_extra - 0.5 * \
            jnp.exp(log_v_extra) * eq_gap  # log likelihood of equation

        log_joint = log_prior + log_boundary_ll * self.llk_weight + eq_ll
        return -log_joint

    @partial(jit, static_argnums=(0, ))
    def step_extra(self, params_extra, opt_state, key):
        # loss = self.loss_extra(params_extra, key)
        loss, d_params = jax.value_and_grad(self.loss_extra)(params_extra, key)
        updates, opt_state = self.optimizer_extra.update(
            d_params, opt_state, params_extra)

        params_extra = optax.apply_updates(params_extra, updates)
        return params_extra, opt_state, loss

    @partial(jit, static_argnums=(0, ))
    def preds_extra(self, params_extra, Xte):

        preds, _ = self.preds(self.params, Xte)

        ker_paras = params_extra['kernel_paras']
        u = params_extra['u']

        u = u.sum(axis=1).reshape(-1, 1)  # sum over trick

        x_p = jnp.tile(self.X_con.flatten(), (self.N_con, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        K = self.kernel_matrix_extra.get_kernel_matrix(X1_p, X2_p, ker_paras)
        Kinv_u = jnp.linalg.solve(K, u)

        N_te = Xte.shape[0]
        x_p11 = jnp.tile(Xte.flatten(), (self.N_con, 1)).T
        x_p22 = jnp.tile(self.X_con.flatten(), (N_te, 1)).T
        X1_p2 = x_p11.flatten()
        X2_p2 = jnp.transpose(x_p22).flatten()
        Kmn = vmap(self.cov_func_extra.kappa,
                   (0, 0, None))(X1_p2.flatten(), X2_p2.flatten(),
                                 ker_paras).reshape(N_te, self.N_con)

        preds_extra = jnp.matmul(Kmn, Kinv_u)

        preds_all = preds + preds_extra

        return preds_all, None

    @partial(jit, static_argnums=(0, ))
    def compute_early_stopping_extra(self, params_extra, key):
        """compute the early stopping criteria"""

        K, Kinv_u, u_xx = self.value_and_grad_kernel(self.params, key)

        K_extra, Kinv_u_extra, u_xx_extra = self.value_and_grad_kernel_extra(
            params_extra, key)

        boundary_gap, eq_gap = self.boundary_and_eq_gap_extra(
            self.params['u'], params_extra['u'], u_xx, u_xx_extra)

        criterion = boundary_gap / self.N + eq_gap / self.N_con

        return criterion

    def train(self, nepoch, seed=0):
        key = jax.random.PRNGKey(seed)
        Q = self.trick_paras['Q']  # number of basis functions

        freq_scale = self.trick_paras['freq_scale']

        early_stopping = {'flag': False, 'epoch': self.trick_paras['nepoch']}

        error_increase_count = 0

        params = {
            "log_tau": 0.0,  # inv var for data ll
            "log_v": 0.0,  # inv var for eq likelihood
            "kernel_paras": {
                'log-w': np.log(1 / Q) * np.ones(Q),
                'log-ls': np.zeros(Q),
                'freq': np.linspace(0, 1, Q) * freq_scale,
                'log-w-matern': np.zeros(1),
                'log-ls-matern': np.zeros(1),
            },
            # u value on the collocation points
            "u": self.trick_paras['init_u_trick'](self, self.trick_paras),
        }

        # params['kernel_paras']['freq'][0] = 0.5

        opt_state = self.optimizer.init(params)
        # self.run_lbfgs(params)

        loss_list = []
        err_list = []
        w_list = []
        freq_list = []

        ls_list = []
        epoch_list = []

        min_err = 2.0
        threshold = 1e-3
        print("here")

        self.pred_func = self.preds

        # to be assigned later
        opt_state_extra = None
        params_extra = None

        # for i in tqdm.tqdm(range(nepoch)):

        change_point = int(nepoch * self.trick_paras['change_point'])

        # for i in range(nepoch):
        for i in tqdm.tqdm(range(nepoch)):

            key, sub_key = jax.random.split(key)

            if i <= change_point:
                params, opt_state, loss = self.step(params, opt_state, sub_key)

            else:
                params_extra, opt_state_extra, loss = self.step_extra(
                    params_extra, opt_state_extra, sub_key)

            if i == change_point:

                print('start to train the extra matern kernel')

                self.params = copy.deepcopy(params)

                params_extra = {
                    "log_tau":
                    copy.deepcopy(params['log_tau']),  # inv var for data ll
                    "log_v": 0.0,  # inv var for eq likelihood
                    "kernel_paras": {
                        'log-w': np.zeros(1),
                        'log-ls': np.zeros(1),
                    },
                    "u": np.zeros((self.N_con, 1))
                }

                self.pred_func = self.preds_extra

                opt_state_extra = self.optimizer_extra.init(params_extra)

            # evluating the error with frequency epoch/20, store the loss and error in a list

            if i % (nepoch / 20) == 0:

                # params = params if i <= int(nepoch / 2) else params_extra
                current_params = params if i <= int(
                    change_point) else params_extra
                preds, _ = self.pred_func(current_params, self.Xte)
                err = jnp.linalg.norm(
                    preds.reshape(-1) -
                    self.yte.reshape(-1)) / jnp.linalg.norm(
                        self.yte.reshape(-1))

                if err < min_err:
                    min_err = err
                elif err - min_err > threshold:
                    error_increase_count += 1

                print('loss = %g' % loss)
                print("It ", i, '  loss = %g ' % loss, " Relative L2 error",
                      err, " min error", min_err)

                loss_list.append(np.log(loss) if loss > 1 else loss)
                err_list.append(err)
                w_list.append(np.exp(params['kernel_paras']['log-w']))
                freq_list.append(params['kernel_paras']['freq'])
                ls_list.append(np.exp(params['kernel_paras']['log-ls']))

                epoch_list.append(i)
                """early stopping criteria"""
                criterion = self.compute_early_stopping(params, sub_key)
                print('criterion = %g' % criterion)

                if i > 0 and (criterion < self.trick_paras['tol']
                              or error_increase_count > 7):
                    print('early stop at epoch %d' % (i))
                    early_stopping['flag'] = True
                    early_stopping['epoch'] = i
                    break

        log_dict = {
            'loss_list': loss_list,
            'err_list': err_list,
            'w_list': w_list,
            'freq_list': freq_list,
            'ls_list': ls_list,
            'epoch_list': epoch_list,
        }

        print('finish training ...')

        self.params_extra = copy.deepcopy(params_extra)

        print('gen fig ...')
        # other_paras = '-extra-GP'
        return log_dict, early_stopping, min_err


def get_source_val(u, x_vec, equation_type):

    if equation_type == 'poisson_1d':

        return vmap(grad(grad(u, 0), 0), (0))(x_vec)

    elif equation_type == 'allencahn_1d':

        return vmap(grad(grad(u, 0), 0),
                    (0))(x_vec) + u(x_vec) * (u(x_vec)**2 - 1)


def test(trick_paras):

    # equation
    equation_dict = {
        'poisson_1d-mix_sin':
        lambda x: jnp.sin(x) + 0.1 * jnp.sin(20 * x) + 0.05 * jnp.sin(100 * x),
        'poisson_1d-single_sin':
        lambda x: jnp.sin(100 * x),
        'poisson_1d-sin_cos':
        lambda x: jnp.sin(6 * x) * jnp.cos(100 * x),
        'poisson_1d-x_time_sinx':
        lambda x: x * jnp.sin(200 * x),
        'poisson_1d-x2_add_sinx':
        lambda x: jnp.sin(500 * x) - 2 * (x - 0.5)**2,
        'allencahn_1d-sin_cos':
        lambda x: jnp.sin(6 * x) * jnp.cos(100 * x),
        'allencahn_1d-single_sin':
        lambda x: jnp.sin(100 * x),
    }

    u = equation_dict[trick_paras['equation']]

    M = 300

    scale = trick_paras['scale']

    X_test = np.linspace(0, 1, num=M).reshape(-1, 1) * scale
    Y_test = u(X_test)
    # collocation points
    # N_col = 200
    N_col = trick_paras['N_col']
    X_col = np.linspace(0, 1, num=N_col).reshape(-1, 1) * scale
    Xind = np.array([0, X_col.shape[0] - 1])
    y = jnp.array([u(X_col[Xind[0]]), u(X_col[Xind[1]])]).reshape(-1)

    eq_type = trick_paras['equation'].split('-')[0]
    src_vals = get_source_val(u, X_col.reshape(-1), eq_type)

    err_list = []
    early_stopping_list = []

    start_time = time.time()

    for fold in range(trick_paras['num_fold']):

        print('fold %d training' % fold)

        model_PIGP = GP_solver_1d_extra(
            Xind,
            y,
            X_col,
            src_vals,
            1e-6,
            X_test,
            Y_test,
            trick_paras,
        )

        # use the fold id as the random seed
        # np.random.seed(fold)
        # random.seed(fold)
        log_dict, early_stopping, min_err = model_PIGP.train(
            trick_paras['nepoch'], fold)

        err_list.append(min_err)
        early_stopping_list.append(early_stopping['epoch'])

        if fold == 0:
            # only store paras, and plot for the first fold
            utils.store_model(model_PIGP, log_dict, trick_paras)

    end_time = time.time()

    err_mean = np.mean(err_list)
    err_std = np.std(err_list)
    stop_epoch_mean = np.mean(early_stopping_list)

    err_dict = {
        'mean': err_mean,
        'std': err_std,
        'err_list': err_list,
        'stop_epoch_mean': stop_epoch_mean,
        'used_time': end_time - start_time,
        'avg_time': (end_time - start_time) / trick_paras['num_fold']
    }

    utils.wrirte_log(model_PIGP, err_dict, trick_paras)
    print('finish writing log ...')

def evals(**kwargs):
    
    args = ExpConfig()
    args.parse(kwargs)

    # check the validity of the equation and kernel
    
    assert args.equation in [       
        'poisson_1d-mix_sin',
        'poisson_1d-single_sin',
        'poisson_1d-sin_cos',
        'poisson_1d-x_time_sinx',
        'poisson_1d-x2_add_sinx',
        'allencahn_1d-sin_cos',
        'allencahn_1d-single_sin'
        ]
    
    config_path = "./config/" + args.equation + ".yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config['equation'] = args.equation
    config['init_u_trick'] = init_func.zeros
    config['kernel_extra'] = Matern52_1d # extra GP kernel to speed up the convergence
    
    if config['scale'] == '2pi':
        config['scale'] = 2*np.pi
    else:
        config['scale'] = 1.0
    
    if args.nepoch is not None:
        config['nepoch'] = args.nepoch

    if args.kernel == 'Matern52_Cos_1d':
        config['kernel']  = Matern52_Cos_1d
    elif args.kernel == 'SE_Cos_1d':
        config['kernel']  = SE_Cos_1d
    elif args.kernel == 'Matern52_1d':
        config['kernel']  = Matern52_1d
    elif args.kernel == 'SE_1d':
        config['kernel']  = SE_1d
    else:
        raise Exception('Invalid Kernel')


    print('equation: %s, kernel: %s, freq_scale: %d' %
                (config['equation'], config['kernel'].__name__, config['freq_scale']))
    
    config['other_paras'] = config['other_paras'] + '-Ncol-%d' % config['N_col'] + 'change_point-%.1f' % +config[
                            'change_point'] + '-extra-GP'

    test(config)
    
if __name__ == '__main__':
    
    fire.Fire(evals) 