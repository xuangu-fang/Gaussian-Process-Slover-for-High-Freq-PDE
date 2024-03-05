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


'''GP solver class for 1d dynamics with single kernel,
 now support poisson-1d and allen-cahn-1d'''


class GP_solver_1d_single(object):

    # equation: u_{xx}  = f(x)
    # Xind: the indices of X_col that corresponds to training points, i.e., boundary points
    # y: training outputs
    # Xcol: collocation points
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
        self.Xind = Xind
        self.y = y
        self.X_col = X_col
        self.src_col = src_col
        self.jitter = jitter
        # X is the 1st and the last point in X_col
        self.X_con = X_col
        self.N = self.Xind.shape[0]
        self.N_con = self.X_con.shape[0]

        self.trick_paras = trick_paras

        self.optimizer = optax.adam(learning_rate=trick_paras['lr'])

        self.llk_weight = trick_paras['llk_weight']

        self.cov_func = trick_paras['kernel']()

        self.kernel_matrix = Kernel_matrix(self.jitter, self.cov_func)

        self.Xte = X_test
        self.yte = Y_test

        self.params = None  # to be assugned after training the mixture-GP
        self.pred_func = None  # to be assugned when starting prediction

        self.eq_type = trick_paras['equation'].split('-')[0]
        assert self.eq_type in ['poisson_1d', 'allencahn_1d']

        print('equation is: ', self.trick_paras['equation'])
        print('kernel is:', self.cov_func.__class__.__name__)

    @partial(jit, static_argnums=(0, ))
    def value_and_grad_kernel(self, params, key):
        '''compute the value of the kernel matrix, along with Kinv_u and u_xx'''

        u = params['u']  # function values at the collocation points
        kernel_paras = params['kernel_paras']
        x_p = jnp.tile(self.X_con.flatten(), (self.N_con, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        # only the cov matrix of func vals
        K = self.kernel_matrix.get_kernel_matrix(X1_p, X2_p, kernel_paras)

        Kinv_u = jnp.linalg.solve(K, u)

        K_dxx1 = vmap(self.cov_func.DD_x1_kappa,
                      (0, 0, None))(X1_p, X2_p, kernel_paras).reshape(
                          self.N_con, self.N_con)
        u_xx = jnp.matmul(K_dxx1, Kinv_u)

        return K, Kinv_u, u_xx

    @partial(jit, static_argnums=(0, ))
    def boundary_and_eq_gap(self, u, u_xx):
        """compute the boundary and equation gap, to construct the training loss or computing the early stopping criteria"""
        # boundary
        boundary_gap = jnp.sum(jnp.square(u[self.Xind].reshape(-1) -
                                          self.y.reshape(-1)))
        # equation
        if self.eq_type == 'poisson_1d':

            eq_gap = jnp.sum(jnp.square(
                u_xx.flatten() - self.src_col.flatten()))

        elif self.eq_type == 'allencahn_1d':

            eq_gap = jnp.sum(jnp.square(u_xx.flatten() +
                             (u*(u**2-1)).flatten() - self.src_col.flatten()))

        else:
            raise NotImplementedError

        return boundary_gap, eq_gap

    @partial(jit, static_argnums=(0, ))
    def loss(self, params, key):
        '''compute the loss function'''
        u = params['u']  # function values at the collocation points
        log_tau = params['log_tau']
        log_v = params['log_v']

        K, Kinv_u, u_xx = self.value_and_grad_kernel(params, key)

        boundary_gap, eq_gap = self.boundary_and_eq_gap(u, u_xx)

        # prior
        log_prior = -0.5 * \
            jnp.linalg.slogdet(
                K)[1]*self.trick_paras['logdet'] - 0.5*jnp.sum(u*Kinv_u)

        # boundary
        log_boundary_ll = 0.5 * self.N * log_tau - 0.5 * \
            jnp.exp(
                log_tau) * boundary_gap
        # equation

        eq_ll = 0.5 * self.N_con * log_v - 0.5 * \
            jnp.exp(log_v) * eq_gap

        log_joint = log_prior + log_boundary_ll * self.llk_weight + eq_ll
        return -log_joint

    @partial(jit, static_argnums=(0, ))
    def step(self, params, opt_state, key):
        # loss = self.loss(params, key)
        loss, d_params = jax.value_and_grad(self.loss)(params, key)
        updates, opt_state = self.optimizer.update(d_params, opt_state, params)

        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @partial(jit, static_argnums=(0, ))
    def preds(self, params, Xte):
        ker_paras = params['kernel_paras']
        u = params['u']

        x_p = jnp.tile(self.X_con.flatten(), (self.N_con, 1)).T
        X1_p = x_p.flatten()
        X2_p = jnp.transpose(x_p).flatten()
        K = self.kernel_matrix.get_kernel_matrix(X1_p, X2_p, ker_paras)
        Kinv_u = jnp.linalg.solve(K, u)

        N_te = Xte.shape[0]
        x_p11 = jnp.tile(Xte.flatten(), (self.N_con, 1)).T
        x_p22 = jnp.tile(self.X_con.flatten(), (N_te, 1)).T
        X1_p2 = x_p11.flatten()
        X2_p2 = jnp.transpose(x_p22).flatten()
        Kmn = vmap(self.cov_func.kappa,
                   (0, 0, None))(X1_p2.flatten(), X2_p2.flatten(),
                                 ker_paras).reshape(N_te, self.N_con)
        preds = jnp.matmul(Kmn, Kinv_u)
        return preds, K

    @partial(jit, static_argnums=(0, ))
    def compute_early_stopping(self, params, key):
        """compute the early stopping criteria"""

        K, Kinv_u, u_xx = self.value_and_grad_kernel(params, key)
        boundary_gap, eq_gap = self.boundary_and_eq_gap(params['u'], u_xx)

        criterion = boundary_gap / self.N + eq_gap/self.N_con

        return criterion

    def train(self, nepoch, seed=0):
        key = jax.random.PRNGKey(109)
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
            },
            # u value on the collocation points
            "u": np.zeros((self.N_con, 1))
        }

        # params['kernel_paras']['freq'][0] = 0.5

        opt_state = self.optimizer.init(params)

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

        # for i in range(nepoch):
        for i in tqdm.tqdm(range(nepoch)):

            key, sub_key = jax.random.split(key)

            params, opt_state, loss = self.step(params, opt_state, sub_key)

            # evluating the error with frequency epoch/20, store the loss and error in a list, also verify the criterion for early stopping

            if i % (nepoch / 20) == 0:

                current_params = params
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
                print("It ", i, '  loss = %g ' % loss,
                      " Relative L2 error", err, " min error", min_err)

                loss_list.append(np.log(loss) if loss > 1 else loss)
                err_list.append(err)
                w_list.append(np.exp(params['kernel_paras']['log-w']))
                freq_list.append(params['kernel_paras']['freq'])
                ls_list.append(np.exp(params['kernel_paras']['log-ls']))

                epoch_list.append(i)

                """early stopping criteria"""
                criterion = self.compute_early_stopping(params, sub_key)
                print('criterion = %g' % criterion)

                # if i > 0 and (criterion < self.trick_paras['tol'] or error_increase_count > 7):
                #     print('early stop at epoch %d' % (i))
                #     early_stopping['flag'] = True
                #     early_stopping['epoch'] = i
                #     break

        log_dict = {
            'loss_list': loss_list,
            'err_list': err_list,
            'w_list': w_list,
            'freq_list': freq_list,
            'ls_list': ls_list,
            'epoch_list': epoch_list,
        }

        print('finish training ...')

        self.params = params
        # other_paras = '-extra-GP'
        # other_paras = self.trick_paras[
        #     'other_paras'] + '-change_point-%.2f' % self.trick_paras[
        #         'change_point']
        # utils.make_fig_1d_extra_GP(self, params_extra, log_dict, other_paras)

        return log_dict, early_stopping, min_err


def get_source_val(u, x_vec,  equation_type):

    if equation_type == 'poisson_1d':

        return vmap(grad(grad(u, 0), 0), (0))(x_vec)

    elif equation_type == 'allencahn_1d':

        return vmap(grad(grad(u, 0), 0), (0))(x_vec) + u(x_vec) * (u(x_vec)**2 - 1)


def test(trick_paras):

    # equation
    equation_dict = {
        'poisson_1d-mix_sin':
        lambda x: jnp.sin(x) + 0.1*jnp.sin(20 * x) + 0.05 * jnp.sin(100 * x),
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

        'poisson_1d-x_time_sinx_scale':
        lambda x: x * jnp.sin(200 * x * np.pi),
    }

    u = equation_dict[trick_paras['equation']]

    M = 300

    x_scale = trick_paras['x_scale']

    X_test = np.linspace(0, 1, num=M).reshape(-1, 1)*x_scale
    Y_test = u(X_test)


    # collocation points

    N_col = trick_paras['N_col']

    X_col = np.linspace(0, 1, num=N_col).reshape(-1, 1)*x_scale

    Xind = np.array([0, X_col.shape[0] - 1])
    y = jnp.array([u(X_col[Xind[0]]), u(X_col[Xind[1]])]).reshape(-1)

    eq_type = trick_paras['equation'].split('-')[0]
    src_vals = get_source_val(u, X_col.reshape(-1), eq_type)

    err_list = []
    early_stopping_list = []

    start_time = time.time()

    for fold in range(trick_paras['num_fold']):

            print('fold %d training' % fold)

            model_PIGP = GP_solver_1d_single(Xind, y, X_col, src_vals, 1e-6, X_test, Y_test,
                                            trick_paras,)

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

    err_dict = {'mean': err_mean, 'std': err_std,
                'err_list': err_list, 'stop_epoch_mean': stop_epoch_mean, 'used_time': end_time - start_time, 'avg_time': (end_time - start_time) / trick_paras['num_fold']}

    utils.wrirte_log(model_PIGP, err_dict, trick_paras)
    print('finish writing log ...')


if __name__ == '__main__':

    equation_list = [
        # 'poisson_1d-mix_sin',
        # 'poisson_1d-single_sin',
        'poisson_1d-sin_cos',
        # 'poisson_1d-x_time_sinx',
        # 'poisson_1d-x2_add_sinx',
        # 'allencahn_1d-sin_cos',
        # 'allencahn_1d-single_sin'
    ]

    kernel_list = [
        Matern52_Cos_1d,
        SE_Cos_1d,
        # Matern52_1d,
        # SE_1d,
    ]

    N_col_list = [300, 200, 100 ,50, 10]  

    for N_col in N_col_list:

        for equation in equation_list:
            for kernel in kernel_list:

                if equation == 'poisson_1d-x2_add_sinx':
                    freq_scale = 100
                elif equation == 'poisson_1d-x_time_sinx':
                    freq_scale = 50
                else:
                    # freq_scale = 30
                    freq_scale = 20

                # uglg code for multi-reso testing
                if equation == 'poisson_1d-x2_add_sinx':
                    x_scale = 1
                    other_paras = '-x-1'
                else:
                    x_scale = 2*np.pi
                    other_paras = '-x-2pi'


                print('equation: %s, kernel: %s, freq_scale: %d' %
                    (equation, kernel.__name__, freq_scale))

                trick_paras = {
                    'equation': equation,
                    'init_u_trick': init_func.zeros,
                    'num_u_trick': 1,
                    'Q': 30,
                    'lr': 1e-2,
                    'llk_weight': 200.0,
                    'kernel': kernel,
                    'kernel_extra': None,
                    'nepoch': 500000,
                    'freq_scale': freq_scale,
                    'logdet': True,
                    'num_fold': 1,
                    'tol': 1e-2,
                    'other_paras': other_paras,
                    'x_scale': x_scale,
                    'N_col': N_col,

                }

                trick_paras['other_paras'] = trick_paras['other_paras'] + \
                    '-Ncol-%d' % trick_paras['N_col']  \
                            + 'multi-reso-test'

                test(trick_paras)
