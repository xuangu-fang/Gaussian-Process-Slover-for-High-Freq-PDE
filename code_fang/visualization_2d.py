import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import os
import pickle
import utils


equation_dict = {
    'poisson_2d-sin_cos': 'u = sin(6x)cos(100x)',
    'poisson_2d-sin_add_cos': 'u = sin(500x)-2*(x-0.5)^2',
    'allencahn_2d-mix-sincos': 'u= sin(x) + 0.1*sin(20x) + 0.05*cos(100x)',
    'advection-sin': 'u = sin(x-200t)'
}


def load_and_predict_2d(prefix, save_name):

    with open(prefix + save_name + '.pkl', 'rb') as f:
        params, log_dict, trick_paras = pickle.load(f)

    model, preds = utils.get_model_2d(params, trick_paras)

    return model, preds, log_dict


def load_and_predict_advection(prefix, save_name):

    with open(prefix + save_name + '.pkl', 'rb') as f:
        params, log_dict, trick_paras = pickle.load(f)

    model, preds = utils.get_model_2d_advection(params, trick_paras)

    return model, preds, log_dict


def draw_fig(model, preds, log_dict):

    err = min(log_dict['err_list'])
    equation_name = model.trick_paras['equation']
    kernel_name = model.cov_func.__class__.__name__

    # generate new figure
    plt.figure(figsize=(6, 6))

    plt.imshow(preds, cmap="hot")

    # print eq name, kernel_name and err as title
    plt.title('Equation: {}, \n Kernel: {},   L2 Err: {:.2e}'.format(
        equation_dict[equation_name], kernel_name, err))

    # to save figure
    prefix = '../figs/' + equation_name + '/' + \
        'Q-%d' % (model.trick_paras['Q']) + '/'

    # create folder if not exist
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # save as pdf and png
    plt.savefig(prefix + kernel_name + '.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(prefix + kernel_name + '.png', dpi=300, bbox_inches='tight')


# kernel_list = ['Matern52_Cos_1d', 'Matern52_1d', 'SE_1d', 'SE_Cos_1d']

# # prefix = '../result_log/poisson_1d-sin_cos/kernel_Matern52_Cos_1d/epoch_1000000/Q30/'
# save_name = 'llk_weight-500.0-nu-1-Q-30-epoch-1000000-lr-0.0100-freqscale=40-logdet-1beta-200'

# for kernel_name in kernel_list:
#     prefix = './result_log/advection-sin/kernel_%s/epoch_1000000/Q30/' % (
#         kernel_name)

#     model, preds, log_dict = load_and_predict_advection(prefix, save_name)
#     draw_fig(model, preds, log_dict)

prefix = './result_log/allencahn_2d-mix-sincos/kernel_Matern52_Cos_1d/epoch_1000000/Q30/'
