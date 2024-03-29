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

import yaml
import fire
# from infras.misc import create_path
from infras.exp_config import ExpConfig


'''GP solver class for 1d advection-like dynamics (only u_x, u_y invloved in the equation) with single kernel'''


class GP_solver_2d_single_advection(object):

    # advection 1d: beta * u_x + u_y= 0
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
        self.bvals = bvals  # Nb dim
        self.X_col = X_col
        self.jitter = jitter
        self.Nb = bvals.size  # number of boundary points
        self.N1 = X_col[0].size
        self.N2 = X_col[1].size
        self.Nc = self.N1 * self.N2
        self.src_vals = src_vals  # N1 by N2

        self.trick_paras = trick_paras

        self.optimizer = optax.adam(learning_rate=trick_paras['lr'])

        self.llk_weight = trick_paras['llk_weight']

        self.cov_func = trick_paras['kernel']()

        self.kernel_matrix = Kernel_matrix(self.jitter, self.cov_func)

        self.Xte = X_test
        self.ute = u_test

        self.beta = trick_paras['beta']

        self.params = None  # to be assugned after training the mixture-GP
        self.pred_func = None  # to be assugned when starting prediction

        self.x_pos_tr_mesh, self.x_pos_tr_mesh_T = np.meshgrid(self.X_col[0],
                                                               self.X_col[0],
                                                               indexing='ij')
        self.y_pos_tr_mesh, self.y_pos_tr_mesh_T = np.meshgrid(self.X_col[1],
                                                               self.X_col[1],
                                                               indexing='ij')

        self.eq_type = trick_paras['equation'].split('-')[0]
        assert self.eq_type in ['advection']

        print('equation is: ', self.trick_paras['equation'])
        print('kernel is:', self.cov_func.__class__.__name__)

    @partial(jit, static_argnums=(0, ))
    def value_and_grad_kernel(self, params, key):
        '''compute the value of the kernel matrix (K1, K2), along with K1inv_u, K2inv_u and u_x, u_y'''

        U = params['U']  # function values at the collocation points, N1 X N2
        kernel_paras_x = params[
            'kernel_paras_1']  # ker params for 1st dimension
        kernel_paras_y = params[
            'kernel_paras_2']  # ker params for 2nd dimension

        K1 = self.kernel_matrix.get_kernel_matrix(self.x_pos_tr_mesh,
                                                  self.x_pos_tr_mesh_T,
                                                  kernel_paras_x)  # N1 x N1
        K2 = self.kernel_matrix.get_kernel_matrix(self.y_pos_tr_mesh,
                                                  self.y_pos_tr_mesh_T,
                                                  kernel_paras_y)  # N2 x N2

        K1inv_U = jnp.linalg.solve(K1, U)  # N1 x N2
        K2inv_Ut = jnp.linalg.solve(K2, U.T)  # N2 x N1

        K_dx1 = vmap(self.cov_func.D_x1_kappa,
                     (0, 0, None))(self.x_pos_tr_mesh.reshape(-1),
                                   self.x_pos_tr_mesh_T.reshape(-1),
                                   kernel_paras_x).reshape(self.N1, self.N1)

        U_x = jnp.matmul(K_dx1, K1inv_U)

        K_dy1 = vmap(self.cov_func.D_x1_kappa,
                     (0, 0, None))(self.y_pos_tr_mesh.reshape(-1),
                                   self.y_pos_tr_mesh_T.reshape(-1),
                                   kernel_paras_y).reshape(self.N2, self.N2)

        U_y = jnp.matmul(K_dy1, K2inv_Ut).T

        return K1, K2, K1inv_U, K2inv_Ut, U_x, U_y

    @partial(jit, static_argnums=(0, ))
    def boundary_and_eq_gap(self, U, U_x, U_y):
        """compute the boundary and equation gap, to construct the training loss or computing the early stopping criteria"""
        # boundary

        u_b = jnp.hstack((U[0, :], U[-1, :], U[:, 0], U[:, -1]))
        boundary_gap = jnp.sum(
            jnp.square(u_b.reshape(-1) - self.bvals.reshape(-1)))
        # equation
        if self.eq_type == 'advection':

            eq_gap = jnp.sum(jnp.square(self.beta * U_x + U_y - self.src_vals))

        else:
            raise NotImplementedError

        return boundary_gap, eq_gap

    @partial(jit, static_argnums=(0, ))
    def loss(self, params, key):
        '''compute the loss function'''
        U = params['U']  # function values at the collocation points
        log_tau = params['log_tau']
        log_v = params['log_v']

        K1, K2, K1inv_U, K2inv_Ut, U_x, U_y = self.value_and_grad_kernel(
            params, key)

        boundary_gap, eq_gap = self.boundary_and_eq_gap(U, U_x, U_y)

        # prior
        log_prior = -0.5 * self.N2 * jnp.linalg.slogdet(
            K1
        )[1] * self.trick_paras['logdet'] - 0.5 * self.N1 * jnp.linalg.slogdet(
            K2)[1] * self.trick_paras['logdet'] - 0.5 * jnp.sum(
                K1inv_U * K2inv_Ut.T)

        # boundary
        log_boundary_ll = 0.5 * self.Nb * log_tau - 0.5 * \
            jnp.exp(
                log_tau) * boundary_gap
        # equation

        eq_ll = 0.5 * self.Nc * log_v - 0.5 * \
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
    def preds(self, params):
        ker_paras_x = params['kernel_paras_1']
        ker_paras_y = params['kernel_paras_2']
        U = params['U']
        K1 = self.kernel_matrix.get_kernel_matrix(self.x_pos_tr_mesh,
                                                  self.x_pos_tr_mesh_T,
                                                  ker_paras_x)
        K1inv_U = jnp.linalg.solve(K1, U)  # N1 x N2

        x_te_cross_mh, x_tr_cross_mh = np.meshgrid(self.Xte[0],
                                                   self.X_col[0],
                                                   indexing='ij')
        Kmn = vmap(self.cov_func.kappa,
                   (0, 0, None))(x_te_cross_mh.reshape(-1),
                                 x_tr_cross_mh.reshape(-1),
                                 ker_paras_x).reshape(self.Xte[0].size,
                                                      self.N1)

        M1 = jnp.matmul(Kmn, K1inv_U)

        K2 = self.kernel_matrix.get_kernel_matrix(self.y_pos_tr_mesh,
                                                  self.y_pos_tr_mesh_T,
                                                  ker_paras_y)
        M2 = jnp.linalg.solve(K2, M1.T)

        y_te_cross_mh, y_tr_cross_mh = np.meshgrid(self.Xte[1],
                                                   self.X_col[1],
                                                   indexing='ij')
        Kmn2 = vmap(self.cov_func.kappa,
                    (0, 0, None))(y_te_cross_mh.reshape(-1),
                                  y_tr_cross_mh.reshape(-1),
                                  ker_paras_y).reshape(self.Xte[1].size,
                                                       self.N2)
        U_pred = jnp.matmul(Kmn2, M2).T
        return U_pred, None

    @partial(jit, static_argnums=(0, ))
    def compute_early_stopping(self, params, key):
        """compute the early stopping criteria"""

        K1, K2, K1inv_U, K2inv_Ut, U_x, U_y = self.value_and_grad_kernel(
            params, key)
        boundary_gap, eq_gap = self.boundary_and_eq_gap(
            params['U'], U_x, U_y)

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
            # "kernel_paras": {'log-w': np.zeros(Q), 'log-ls': np.zeros(Q), 'freq': np.linspace(0, 1, Q)*100},
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

        # for i in range(nepoch):
        for i in tqdm.tqdm(range(nepoch)):

            key, sub_key = jax.random.split(key)

            params, opt_state, loss = self.step(params, opt_state, sub_key)

            # evluating the error with frequency epoch/20, store the loss and error in a list, also verify the criterion for early stopping

            if i % (nepoch / 20) == 0:

                current_params = params
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
                criterion = self.compute_early_stopping(params, sub_key)
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


def get_source_val(u, x_pos, y_pos, equation_type, beta):
    x_mesh, y_mesh = np.meshgrid(x_pos, y_pos, indexing='ij')
    x_vec = x_mesh.reshape(-1)
    y_vec = y_mesh.reshape(-1)

    if equation_type == 'advection':
        return beta*(vmap(grad(u, 0), (0, 0))(x_vec, y_vec)) + vmap(grad(u, 1), (0, 0))(x_vec, y_vec)
    else:
        raise NotImplementedError


def get_mesh_data(u, M1, M2,scale):
    x_coor = np.linspace(0, 1, num=M1)*scale
    y_coor = np.linspace(0, 1, num=M2)*scale
    x_mesh, y_mesh = np.meshgrid(x_coor, y_coor, indexing='ij')
    u_mesh = u(x_mesh, y_mesh)
    return x_coor, y_coor, u_mesh


def get_boundary_vals(u_mesh):
    return jnp.hstack((u_mesh[0, :], u_mesh[-1, :], u_mesh[:, 0], u_mesh[:,
                                                                         -1]))

def get_boundary_vals_only_init(u_mesh):
    """use init conditions (t=0) only"""
    return jnp.hstack((u_mesh[:, 0],))

def test(trick_paras):

    # equation
    beta = trick_paras['beta']
    equation_dict = {
        'advection-sin':
        lambda x, y: jnp.sin(x-beta*y),
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
    # bvals = get_boundary_vals_only_init(u_mh)
    
    bvals = get_boundary_vals(u_mh)


    src_vals = get_source_val(u, x_pos_tr, y_pos_tr, eq_type, beta)
    src_vals = src_vals.reshape((x_pos_tr.size, y_pos_tr.size))
    X_test = (x_pos_test, y_pos_test)
    u_test = u_test_mh
    X_col = (x_pos_tr, y_pos_tr)

    err_list = []
    early_stopping_list = []

    start_time = time.time()

    for fold in range(trick_paras['num_fold']):

        print('fold %d training' % fold)

        model_PIGP = GP_solver_2d_single_advection(
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





def evals(**kwargs):
    
    args = ExpConfig()
    args.parse(kwargs)

    # check the validity of the equation and kernel
    
    assert args.equation in [ 'advection-sin']
    
    config_path = "./config/" + args.equation + ".yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config['equation'] = args.equation
    config['init_u_trick'] = init_func.zeros
    config['kernel_extra'] = None
    
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
    
    config['other_paras'] = config['other_paras'] +'-beta-%d' %  config['beta']+ '-Ncol-%d' % config['N_col']

    test(config)
    
if __name__ == '__main__':
    
    fire.Fire(evals) 