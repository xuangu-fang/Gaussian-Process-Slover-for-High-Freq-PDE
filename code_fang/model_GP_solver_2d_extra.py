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
from model_GP_solver_2d import GP_solver_2d_single

'''GP solver for 2d equation with a extra GP, which can accelerate the convergence by capturing the low frequency part of the solution quickly'''


class GP_solver_2d_extra(GP_solver_2d_single):

    # equation: u_{xx} + u_{yy}  = f(x,y,u)
    # bvals: 1d array, boundary values
    # X_col = (x_pos, y_pos), x_pos: 1d array, y_pos: 1d array
    # src_vals: source values at the collocation mesh, N1 x N2
    # X_test = (x_test_pos, y_test_pos): x_test_pos: 1d array, y_test_pos: 1d array
    # u_test:  M1 x M2
    def __init__(self,
                 bvals,
                 X_col,
                 src_vals,
                 jitter,
                 X_test,
                 u_test,
                 trick_paras=None,
                 fix_dict=None):
        
        super(GP_solver_2d_extra, self).__init__(bvals, X_col, src_vals, jitter, X_test, u_test, trick_paras, fix_dict)

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
        '''compute the value of the kernel matrix (K1, K2), along with K1inv_u, K2inv_u and u_xx, u_yy'''

        U = params_extra['U']  # function values at the collocation points, N1 X N2
        kernel_paras_x = params_extra[
            'kernel_paras_1']  # ker params for 1st dimension
        kernel_paras_y = params_extra[
            'kernel_paras_2']  # ker params for 2nd dimension

        K1 = self.kernel_matrix_extra.get_kernel_matrix(self.x_pos_tr_mesh,
                                                  self.x_pos_tr_mesh_T,
                                                  kernel_paras_x)  # N1 x N1
        K2 = self.kernel_matrix_extra.get_kernel_matrix(self.y_pos_tr_mesh,
                                                  self.y_pos_tr_mesh_T,
                                                  kernel_paras_y)  # N2 x N2

        K1inv_U = jnp.linalg.solve(K1, U)  # N1 x N2
        K2inv_Ut = jnp.linalg.solve(K2, U.T)  # N2 x N1

        K_dxx1 = vmap(self.cov_func_extra.DD_x1_kappa,
                      (0, 0, None))(self.x_pos_tr_mesh.reshape(-1),
                                    self.x_pos_tr_mesh_T.reshape(-1),
                                    kernel_paras_x).reshape(self.N1, self.N1)

        U_xx = jnp.matmul(K_dxx1, K1inv_U)

        K_dyy1 = vmap(self.cov_func_extra.DD_x1_kappa,
                      (0, 0, None))(self.y_pos_tr_mesh.reshape(-1),
                                    self.y_pos_tr_mesh_T.reshape(-1),
                                    kernel_paras_y).reshape(self.N2, self.N2)

        U_yy = jnp.matmul(K_dyy1, K2inv_Ut).T

        return K1, K2, K1inv_U, K2inv_Ut, U_xx, U_yy

    @partial(jit, static_argnums=(0, ))
    def boundary_and_eq_gap_extra(self, U, U_xx, U_yy, U_extra, U_xx_extra, U_yy_extra):
        """compute the boundary and equation gap, to construct the training loss or computing the early stopping criteria"""
        # boundary
        u_b = jnp.hstack((U[0, :], U[-1, :], U[:, 0], U[:, -1]))
        u_b_extra = jnp.hstack((U_extra[0, :], U_extra[-1, :], U_extra[:, 0], U_extra[:, -1]))


        boundary_gap = jnp.sum(
            jnp.square(u_b.reshape(-1) + u_b_extra.reshape(-1) - self.bvals.reshape(-1)))
        
        # equation
        if self.eq_type == 'poisson_2d':

            eq_gap = jnp.sum(jnp.square(U_xx + U_yy + U_xx_extra + U_yy_extra - self.src_vals))

        elif self.eq_type == 'allencahn_2d':

            U = U + U_extra
            eq_gap = jnp.sum(
                jnp.square(U_xx + U_yy  + U_xx_extra + U_yy_extra + U * (U**2 - 1) - self.src_vals))

        else:
            raise NotImplementedError

        return boundary_gap, eq_gap

    @partial(jit, static_argnums=(0, ))
    def loss_extra(self, params_extra, key):
        '''compute the loss function'''
        U = self.params['U']  # function values at the collocation points

        K1, K2, K1inv_U, K2inv_Ut, U_xx, U_yy = self.value_and_grad_kernel(
            self.params, key)
        
        U_extra = params_extra['U']  # function values at the collocation points
        log_v_extra = params_extra['log_v']
        log_tau_extra = params_extra['log_tau']

        K1_extra, K2_extra, K1inv_U_extra, K2inv_Ut_extra, U_xx_extra, U_yy_extra = self.value_and_grad_kernel_extra(
            params_extra, key)


        boundary_gap, eq_gap = self.boundary_and_eq_gap_extra(
            U, U_xx, U_yy, U_extra, U_xx_extra, U_yy_extra)

        # prior
        log_prior = -0.5 * self.N2 * jnp.linalg.slogdet(
            K1_extra
        )[1] * self.trick_paras['logdet'] - 0.5 * self.N1 * jnp.linalg.slogdet(
            K2_extra)[1] * self.trick_paras['logdet'] - 0.5 * jnp.sum(
                K1inv_U_extra * K2inv_Ut_extra.T)

        # boundary
        log_boundary_ll = 0.5 * self.Nb * log_tau_extra - 0.5 * \
            jnp.exp(
                log_tau_extra) * boundary_gap
        # equation

        eq_ll = 0.5 * self.Nc * log_v_extra - 0.5 * \
            jnp.exp(log_v_extra) * eq_gap

        log_joint = log_prior + log_boundary_ll * self.llk_weight + eq_ll
        return -log_joint

    @partial(jit, static_argnums=(0, ))
    def step_extra(self, params_extra, opt_state, key):
        # loss = self.loss(params, key)
        loss, d_params = jax.value_and_grad(self.loss_extra)(params_extra, key)
        updates, opt_state = self.optimizer_extra.update(d_params, opt_state, params_extra)

        params_extra = optax.apply_updates(params_extra, updates)
        return params_extra, opt_state, loss

    @partial(jit, static_argnums=(0, ))
    def preds_extra(self, params_extra):

        preds, _ = self.preds(self.params)

        ker_paras_x = params_extra['kernel_paras_1']
        ker_paras_y = params_extra['kernel_paras_2']
        U = params_extra['U']
        K1 = self.kernel_matrix_extra.get_kernel_matrix(self.x_pos_tr_mesh,
                                                  self.x_pos_tr_mesh_T,
                                                  ker_paras_x)
        K1inv_U = jnp.linalg.solve(K1, U)  # N1 x N2

        x_te_cross_mh, x_tr_cross_mh = np.meshgrid(self.Xte[0],
                                                   self.X_col[0],
                                                   indexing='ij')
        Kmn = vmap(self.cov_func_extra.kappa,
                   (0, 0, None))(x_te_cross_mh.reshape(-1),
                                 x_tr_cross_mh.reshape(-1),
                                 ker_paras_x).reshape(self.Xte[0].size,
                                                      self.N1)

        M1 = jnp.matmul(Kmn, K1inv_U)

        K2 = self.kernel_matrix_extra.get_kernel_matrix(self.y_pos_tr_mesh,
                                                  self.y_pos_tr_mesh_T,
                                                  ker_paras_y)
        M2 = jnp.linalg.solve(K2, M1.T)

        y_te_cross_mh, y_tr_cross_mh = np.meshgrid(self.Xte[1],
                                                   self.X_col[1],
                                                   indexing='ij')
        Kmn2 = vmap(self.cov_func_extra.kappa,
                    (0, 0, None))(y_te_cross_mh.reshape(-1),
                                  y_tr_cross_mh.reshape(-1),
                                  ker_paras_y).reshape(self.Xte[1].size,
                                                       self.N2)
        U_pred_extra = jnp.matmul(Kmn2, M2).T

        U_preds_all = preds + U_pred_extra
        return U_preds_all, None

    @partial(jit, static_argnums=(0, ))
    def compute_early_stopping_extra(self, params_extra, key):
        """compute the early stopping criteria"""

        K1, K2, K1inv_U, K2inv_Ut, U_xx, U_yy = self.value_and_grad_kernel(
            self.params, key)
        
        K1_extra, K2_extra, K1inv_U_extra, K2inv_Ut_extra, U_xx_extra, U_yy_extra = self.value_and_grad_kernel_extra(
            params_extra, key)

        boundary_gap, eq_gap = self.boundary_and_eq_gap_extra(
            self.params['U'], U_xx, U_yy, params_extra['U'], U_xx_extra, U_yy_extra)

        criterion = boundary_gap / self.Nb + eq_gap / self.Nc

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
            "kernel_paras_1": {
                'log-w': np.log(1 / Q) * np.ones(Q),
                'log-ls': np.zeros(Q),
                'freq': np.linspace(0, 1, Q) * freq_scale,
            },
            "kernel_paras_2": {
                'log-w': np.log(1 / Q) * np.ones(Q),
                'log-ls': np.zeros(Q),
                'freq': np.linspace(0, 1, Q) * freq_scale,
            },
            # u value on the collocation points
            "U": np.zeros((self.N1, self.N2)),
        }

        opt_state = self.optimizer.init(params)

        loss_list = []
        err_list = []

        w_list_k1 = []
        freq_list_k1 = []
        ls_list_k1 = []

        w_list_k2 = []
        freq_list_k2 = []
        ls_list_k2 = []

        epoch_list = []

        min_err = 2.0
        threshold = 1e-3
        print("here")

        self.pred_func = self.preds

        # to be assigned later
        opt_state_extra = None
        params_extra = None

        change_point = int(nepoch * self.trick_paras['change_point'])
        print('change point is {}'.format(change_point))

        # for i in range(nepoch):
        for i in tqdm.tqdm(range(nepoch)):

            key, sub_key = jax.random.split(key)

            if i <= change_point:
                params, opt_state, loss = self.step(params, opt_state, sub_key)

            else:
                params_extra, opt_state_extra, loss = self.step_extra(
                    params_extra, opt_state_extra, sub_key)
                
            if i == change_point:

                print('start to train the matern kernel')
                self.params = copy.deepcopy(params)

                params_extra = {
                    "log_tau": 0.0,  # inv var for data ll
                    "log_v": 0.0,  # inv var for eq likelihood
                    "kernel_paras_1": {
                        'log-w': np.zeros(1),
                        'log-ls': np.zeros(1),
                    },
                    "kernel_paras_2": {
                        'log-w': np.zeros(1),
                        'log-ls': np.zeros(1),
                    },
                    # u value on the collocation points
                    "U": np.zeros((self.N1, self.N2)),
                }

                self.pred_func = self.preds_extra

                opt_state_extra = self.optimizer_extra.init(params_extra)

            # evluating the error with frequency epoch/20, store the loss and error in a list, also verify the criterion for early stopping

            if i % (nepoch / 20) == 0:

                current_params = params if i <= int(
                    change_point) else params_extra
                
                preds, _ = self.pred_func(current_params)

                err = jnp.linalg.norm(
                    preds.reshape(-1) -
                    self.ute.reshape(-1)) / jnp.linalg.norm(
                        self.ute.reshape(-1))

                if err < min_err:
                    min_err = err
                elif err - min_err > threshold:
                    error_increase_count += 1

                print('loss = %g' % loss)
                print("It ", i, '  loss = %g ' % loss, " Relative L2 error",
                      err, " min error", min_err)

                loss_list.append(np.log(loss) if loss > 1 else loss)
                err_list.append(err)

                w_list_k1.append(np.exp(params['kernel_paras_1']['log-w']))
                freq_list_k1.append(params['kernel_paras_1']['freq'])
                ls_list_k1.append(np.exp(params['kernel_paras_1']['log-ls']))

                w_list_k2.append(np.exp(params['kernel_paras_2']['log-w']))
                freq_list_k2.append(params['kernel_paras_2']['freq'])
                ls_list_k2.append(np.exp(params['kernel_paras_2']['log-ls']))

                epoch_list.append(i)
                """early stopping criteria"""

                if i < change_point:
                    criterion = self.compute_early_stopping(params, sub_key)

                else:
                    criterion = self.compute_early_stopping_extra(params_extra,sub_key)

                print('criterion = %g' % criterion)

                # if i > 0 and (criterion < self.trick_paras['tol']
                #               or error_increase_count > 7):
                #     print('early stop at epoch %d' % (i))
                #     early_stopping['flag'] = True
                #     early_stopping['epoch'] = i
                #     break

        log_dict = {
            'loss_list': loss_list,
            'err_list': err_list,
            'w_list_k1': w_list_k1,
            'freq_list_k1': freq_list_k1,
            'ls_list_k1': ls_list_k1,
            'w_list_k2': w_list_k2,
            'freq_list_k2': freq_list_k2,
            'ls_list_k2': ls_list_k2,
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


def get_source_val(u, x_pos, y_pos, equation_type):
    x_mesh, y_mesh = np.meshgrid(x_pos, y_pos, indexing='ij')
    x_vec = x_mesh.reshape(-1)
    y_vec = y_mesh.reshape(-1)

    if equation_type == 'poisson_2d':
        return vmap(grad(grad(u, 0), 0), (0, 0))(x_vec, y_vec) + \
            vmap(grad(grad(u, 1), 1), (0, 0))(x_vec, y_vec)
    elif equation_type == 'allencahn_2d':
        return vmap(grad(grad(u, 0), 0), (0, 0))(x_vec, y_vec) + \
            vmap(grad(grad(u, 1), 1), (0, 0))(x_vec, y_vec) + \
            u(x_vec, y_vec)*(u(x_vec, y_vec)**2-1)


def get_mesh_data(u, M1, M2,scale):
    x_coor = np.linspace(0, 1, num=M1)*scale
    y_coor = np.linspace(0, 1, num=M2)*scale
    x_mesh, y_mesh = np.meshgrid(x_coor, y_coor, indexing='ij')
    u_mesh = u(x_mesh, y_mesh)
    return x_coor, y_coor, u_mesh


def get_boundary_vals(u_mesh):
    return jnp.hstack((u_mesh[0, :], u_mesh[-1, :], u_mesh[:, 0], u_mesh[:,
                                                                         -1]))


def test(trick_paras):

    # equation
    equation_dict = {
        'poisson_2d-sin_sin':
        lambda x, y: jnp.sin(100 * x) * jnp.sin(100 * y),
        'poisson_2d-sin_cos':
        lambda x, y: jnp.sin(100 * x) * jnp.cos(100 * y),
        'poisson_2d-sin_add_cos':
        lambda x, y: jnp.sin(6 * x) * jnp.cos(20 * x) + jnp.sin(
            6 * y) * jnp.cos(20 * y),
        'allencahn_2d-mix-sincos':
        lambda x, y: (jnp.sin(x) + 0.1 * jnp.sin(20 * x) + jnp.cos(100 * x)) *
        (jnp.sin(y) + 0.1 * jnp.sin(20 * y) + jnp.cos(100 * y)),
    }

    u = equation_dict[trick_paras['equation']]
    eq_type = trick_paras['equation'].split('-')[0]

    scale = trick_paras['scale']

    M = 300
    x_pos_test, y_pos_test, u_test_mh = get_mesh_data(u, M, M,scale)
    # collocation points  in each dimension
    # N = 200
    N = trick_paras['N_col']

    x_pos_tr, y_pos_tr, u_mh = get_mesh_data(u, N, N,scale)
    bvals = get_boundary_vals(u_mh)

    src_vals = get_source_val(u, x_pos_tr, y_pos_tr, eq_type)
    src_vals = src_vals.reshape((x_pos_tr.size, y_pos_tr.size))
    X_test = (x_pos_test, y_pos_test)
    u_test = u_test_mh
    X_col = (x_pos_tr, y_pos_tr)

    err_list = []
    early_stopping_list = []

    start_time = time.time()

    for fold in range(trick_paras['num_fold']):

        print('fold %d training' % fold)

        model_PIGP = GP_solver_2d_extra(
            bvals,
            X_col,
            src_vals,
            1e-6,
            X_test,
            u_test,
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


if __name__ == '__main__':

    equation_list = [
        #  'poisson_2d-sin_sin',
        # 'poisson_2d-sin_cos',
        # 'poisson_2d-sin_add_cos',
        'allencahn_2d-mix-sincos',
    ]

    kernel_list = [
        Matern52_Cos_1d,
        SE_Cos_1d,
        Matern52_1d,
        SE_1d,
    ]
    change_point_list = [0.5]

    for change_point in change_point_list:
        for equation in equation_list:
            for kernel in kernel_list:

                freq_scale = 20

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
                    'kernel_extra': Matern52_1d,
                    'nepoch': 10000,
                    'freq_scale': freq_scale,
                    'logdet': True,
                    'num_fold': 1,
                    'tol': 1e-2,
                    'other_paras': 'extra_GP-x-2pi',
                    'change_point': change_point,
                    'scale':2*np.pi,
                    'N_col': 400,
                }

                trick_paras['other_paras'] = trick_paras[
                    'other_paras'] + 'change_point-%.1f' % +trick_paras[
                        'change_point'] + '-Ncol-%d' % trick_paras['N_col']
                
                test(trick_paras)
