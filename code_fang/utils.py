import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import os
import pickle
import model_GP_solver_1d
import model_GP_solver_2d
import model_GP_solver_advection
import model_GP_solver_1d_extra


def identity(x):
    return x


def square_norm(x):
    return x**2 / jnp.sum(x**2)


def soft_max(x):
    return jnp.exp(x) / jnp.sum(jnp.exp(x))


def save_paras(model, params, log_dict, other_paras=''):

    Q = model.trick_paras['Q']
    nepoch = model.trick_paras['nepoch']
    prefix = 'result_analysis/' + model.trick_paras['equation'] + '/kernel_' + \
        model.cov_func.__class__.__name__ + \
        '/epoch_'+str(nepoch)+'/Q'+str(Q)+'/'

    # build the folder if not exist
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    fig_name = 'llk_w-%.1f-' % (model.llk_weight) + model.trick_paras[
        'init_u_trick'].__name__ + '-Q-%d-epoch-%d-lr-%.4f-freqscale=%d' % (
            Q, nepoch, model.trick_paras['lr'],
            model.trick_paras['freq_scale']) + other_paras

    # save the params and log_dict as pickle
    with open(prefix + fig_name + '.pickle', 'wb') as f:
        pickle.dump([params, log_dict], f)


def make_fig_1d(model, params, log_dict, other_paras=''):

    # plot a figure with 6 subplots. 1 for the truth- prediction, 2 for the loss curve, 3 for the error curve, 4,5,6 for scatter of the weights, freq, and ls

    loss_list = log_dict['loss_list']
    err_list = log_dict['err_list']
    epoch_list = log_dict['epoch_list']
    w_list = log_dict['w_list']
    freq_list = log_dict['freq_list']
    ls_list = log_dict['ls_list']

    Q = model.trick_paras['Q']
    nepoch = model.trick_paras['nepoch']
    num_u_trick = model.trick_paras['num_u_trick']

    plt.figure(figsize=(20, 10))

    # first subplot
    plt.subplot(2, 3, 1)
    preds, _ = model.preds(params, model.Xte)
    Xtr = model.X_col[model.Xind]

    plt.plot(model.Xte.flatten(), model.yte.flatten(), 'k-', label='Truth')
    plt.plot(model.Xte.flatten(), preds.flatten(), 'r-', label='Pred')
    plt.scatter(Xtr.flatten(), model.y.flatten(), c='g', label='Train')
    plt.legend(loc=2)
    plt.title('pred-truth:loss = %g, err = %g' % (loss_list[-1], err_list[-1]))

    # second subplot: loss curve, x-axis is the epoch, y-axis is the log-loss
    plt.subplot(2, 3, 2)
    plt.plot(epoch_list, loss_list)
    plt.title('loss curve')

    # third subplot: error curve
    plt.subplot(2, 3, 3)
    plt.plot(epoch_list, err_list)
    plt.title('error curve')

    # fourth subplot: scatter of the weights at each test point, which store on the list w_list, the x-axies is epoch, y-axis is the weights, make the marker size smaller to make the plot clearer

    plt.subplot(2, 3, 4)
    for i in range(Q):
        plt.scatter(epoch_list, [w[i] for w in w_list], s=10)

        # if the weight is significant, label each scatter plot with the corresponding id and freq
        weight = [w[i] for w in w_list][-1]
        if weight > 1e-2:
            plt.text(epoch_list[-1], weight,
                     '%s-th_freq-%.1f' % (str(i), freq_list[-1][i]))

    plt.title('weights scatter')

    # fifth subplot: scatter of the freq at each test point, which store on the list freq_list, the x-axies is epoch, y-axis is the freq
    plt.subplot(2, 3, 5)
    for i in range(Q):
        plt.scatter(epoch_list, [f[i] for f in freq_list], s=10)
    plt.title('freq scatter')

    # sixth subplot: scatter of the ls at each test point, which store on the list ls_list, the x-axies is epoch, y-axis is the ls
    plt.subplot(2, 3, 6)
    for i in range(Q):
        plt.scatter(epoch_list, [l[i] for l in ls_list], s=10)

        # if the ls is significant, label each scatter plot with the corresponding id and freq
        ls = [l[i] for l in ls_list][-1]
        if ls > 1e-2:
            plt.text(epoch_list[-1], ls,
                     '%s-th_freq-%.1f' % (str(i), freq_list[-1][i]))
    plt.title('ls scatter')

    fix_prefix_dict = {1: '_fix_', 0: '_nonfix_'}
    fix_prefix = 'w'+fix_prefix_dict[model.fix_dict['log-w']]+'ls' + \
        fix_prefix_dict[model.fix_dict['log-ls']] + \
        'freq'+fix_prefix_dict[model.fix_dict['freq']]

    # make the whole figure title to be the name of the trick
    plt.suptitle(
        fix_prefix + '\n' + model.trick_paras['init_u_trick'].__name__ +
        '-nU-%d-Q-%d-epoch-%d-lr-%.4f-freqscale-%d' %
        (num_u_trick, Q, nepoch, model.trick_paras['lr'],
         model.trick_paras['freq_scale']), )

    prefix = 'result_analysis/' + model.trick_paras['equation'] + '/kernel_' + \
        model.cov_func.__class__.__name__ + \
        '/epoch_'+str(nepoch)+'/Q'+str(Q)+'/'

    # build the folder if not exist
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    fig_name = 'llk_w-%.1f-' % (model.llk_weight) + model.trick_paras[
        'init_u_trick'].__name__ + '-nU-%d-Q-%d-epoch-%d-lr-%.4f-freqscale=%d-logdet-%d' % (
            num_u_trick, Q, nepoch, model.trick_paras['lr'],
            model.trick_paras['freq_scale'],
            model.trick_paras['logdet']) + other_paras

    print('save fig to ', prefix + fig_name + '.png')

    plt.savefig(prefix + fig_name + '.png')


def make_fig_2d(model, params, log_dict, other_paras=''):
    fontsize = 36
    # plot a figure with 9 subplots. 1 for the truth- prediction, 2 for the loss curve, 3 for the error curve, {4,5,6} for scatter of the weights, freq, and ls for x1, {7,8,9} for scatter of the weights, freq, and ls for x2

    loss_list = log_dict['loss_list']
    err_list = log_dict['err_list']
    epoch_list = log_dict['epoch_list']
    w_list_k1 = log_dict['w_list_k1']
    w_list_k2 = log_dict['w_list_k2']
    freq_list_k1 = log_dict['freq_list_k1']
    freq_list_k2 = log_dict['freq_list_k2']
    ls_list_k1 = log_dict['ls_list_k1']
    ls_list_k2 = log_dict['ls_list_k2']

    Q = model.trick_paras['Q']
    nepoch = model.trick_paras['nepoch']
    num_u_trick = model.trick_paras['num_u_trick']

    plt.figure(figsize=(20, 20))

    # first subplot- pred of 2d possion
    plt.subplot(3, 3, 1)
    preds, _ = model.preds(params)

    plt.imshow(preds, cmap="hot")
    plt.title('pred-2d:loss = %g, err = %g' % (loss_list[-1], err_list[-1]),
              fontsize=fontsize * 0.5)

    # second subplot: ground truth of 2d possion
    plt.subplot(3, 3, 2)
    plt.imshow(model.ute, cmap="hot")
    plt.title('ground-truth-2d', fontsize=fontsize * 0.5)

    # third subplot: error curve
    plt.subplot(3, 3, 3)
    plt.plot(epoch_list, err_list)
    plt.title('error curve', fontsize=fontsize * 0.5)

    # fourth subplot: scatter of the weights at each test point, which store on the list w_list, the x-axies is epoch, y-axis is the weights, make the marker size smaller to make the plot clearer

    plt.subplot(3, 3, 4)
    # for i in range(Q):
    #     plt.scatter(epoch_list, [w[i] for w in w_list_k1], s=10)
    for i in range(Q):
        plt.scatter(epoch_list, [w[i] for w in w_list_k1], s=10)

        # if the weight is significant, label each scatter plot with the corresponding id and freq
        weight = [w[i] for w in w_list_k1][-1]
        if weight > 1e-2:
            plt.text(epoch_list[-1], weight,
                     '%s-th_freq-%.1f' % (str(i), freq_list_k1[-1][i]))
    plt.title('weights scatter-k1', fontsize=fontsize * 0.5)

    # fifth subplot: scatter of the freq at each test point, which store on the list freq_list, the x-axies is epoch, y-axis is the freq
    plt.subplot(3, 3, 5)
    for i in range(Q):
        plt.scatter(epoch_list, [f[i] for f in freq_list_k1], s=10)
    plt.title('freq scatter-k1', fontsize=fontsize * 0.5)

    # sixth subplot: scatter of the ls at each test point, which store on the list ls_list, the x-axies is epoch, y-axis is the ls
    plt.subplot(3, 3, 6)

    for i in range(Q):
        plt.scatter(epoch_list, [l[i] for l in ls_list_k1], s=10)

        # if the ls is significant, label each scatter plot with the corresponding id and freq
        ls = [l[i] for l in ls_list_k1][-1]
        if ls > 1e-2:
            plt.text(epoch_list[-1], ls,
                     '%s-th_freq-%.1f' % (str(i), freq_list_k1[-1][i]))

    plt.title('ls scatter-k1', fontsize=fontsize * 0.5)

    plt.subplot(3, 3, 7)
    for i in range(Q):
        plt.scatter(epoch_list, [w[i] for w in w_list_k2], s=10)

        # if the weight is significant, label each scatter plot with the corresponding id and freq
        weight = [w[i] for w in w_list_k2][-1]
        if weight > 1e-2:
            plt.text(epoch_list[-1], weight,
                     '%s-th_freq-%.1f' % (str(i), freq_list_k2[-1][i]))

    plt.title('weights scatter-k2', fontsize=fontsize * 0.5)

    plt.subplot(3, 3, 8)
    for i in range(Q):
        plt.scatter(epoch_list, [f[i] for f in freq_list_k2], s=10)
    plt.title('freq scatter-k2', fontsize=fontsize * 0.5)

    plt.subplot(3, 3, 9)
    for i in range(Q):
        plt.scatter(epoch_list, [l[i] for l in ls_list_k2], s=10)

        # if the ls is significant, label each scatter plot with the corresponding id and freq
        ls = [l[i] for l in ls_list_k2][-1]
        if ls > 1e-2:
            plt.text(epoch_list[-1], ls,
                     '%s-th_freq-%.1f' % (str(i), freq_list_k2[-1][i]))

    plt.title('ls scatter-k2', fontsize=fontsize * 0.5)

    fix_prefix_dict = {1: '_fix_', 0: '_nonfix_'}
    fix_prefix = 'w'+fix_prefix_dict[model.fix_dict['log-w']]+'ls' + \
        fix_prefix_dict[model.fix_dict['log-ls']] + \
        'freq'+fix_prefix_dict[model.fix_dict['freq']]

    # make the whole figure title to be the name of the trick
    plt.suptitle(fix_prefix + '\n' +
                 model.trick_paras['init_u_trick'].__name__ +
                 '-nU-%d-Q-%d-epoch-%d-lr-%.4f' %
                 (num_u_trick, Q, nepoch, model.trick_paras['lr']),
                 fontsize=fontsize)

    prefix = 'result_analysis/' + model.trick_paras['equation'] + \
        '/kernel_'+model.cov_func.__class__.__name__ + \
        '/epoch_'+str(nepoch)+'/Q'+str(Q)+'/'

    # build the folder if not exist
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    fig_name = fix_prefix + 'llk_w-%.1f-' % (
        model.llk_weight
    ) + model.trick_paras[
        'init_u_trick'].__name__ + '-nu-%d-Q-%d-epoch-%d-lr-%.4f-freqscale=%d-logdet-%d' % (
            num_u_trick, Q, nepoch, model.trick_paras['lr'],
            model.trick_paras['freq_scale'],
            model.trick_paras['logdet']) + other_paras
    print('save fig to ', prefix + fig_name)

    plt.savefig(prefix + fig_name + '.png')


def make_fig_1d_extra_GP(model, params_extra, log_dict, other_paras=''):

    # plot a figure with 6 subplots. 1 for the truth- prediction, 2 for the loss curve, 3 for the error curve, 4,5,6 for scatter of the weights, freq, and ls

    loss_list = log_dict['loss_list']
    err_list = log_dict['err_list']
    epoch_list = log_dict['epoch_list']
    w_list = log_dict['w_list']
    freq_list = log_dict['freq_list']
    ls_list = log_dict['ls_list']

    matern_w_list = log_dict['matern_w_list']
    matern_ls_list = log_dict['matern_ls_list']

    Q = model.trick_paras['Q']
    nepoch = model.trick_paras['nepoch']
    num_u_trick = model.trick_paras['num_u_trick']

    plt.figure(figsize=(20, 20))

    # first subplot
    plt.subplot(3, 3, 1)
    preds, _ = model.pred_func(params_extra, model.Xte)
    Xtr = model.X_col[model.Xind]

    plt.plot(model.Xte.flatten(), model.yte.flatten(), 'k-', label='Truth')
    plt.plot(model.Xte.flatten(), preds.flatten(), 'r-', label='Pred')
    plt.scatter(Xtr.flatten(), model.y.flatten(), c='g', label='Train')
    plt.legend(loc=2)
    plt.title('pred-truth:loss = %g, err = %g' % (loss_list[-1], err_list[-1]))

    # second subplot: loss curve, x-axis is the epoch, y-axis is the log-loss
    plt.subplot(3, 3, 2)
    plt.plot(epoch_list, loss_list)
    plt.title('loss curve')

    # third subplot: error curve
    plt.subplot(3, 3, 3)
    plt.plot(epoch_list, err_list)
    plt.title('error curve')

    # fourth subplot: scatter of the weights at each test point, which store on the list w_list, the x-axies is epoch, y-axis is the weights, make the marker size smaller to make the plot clearer

    plt.subplot(3, 3, 4)
    for i in range(Q):
        plt.scatter(epoch_list, [w[i] for w in w_list], s=10)

        # if the weight is significant, label each scatter plot with the corresponding id and freq
        weight = [w[i] for w in w_list][-1]
        if weight > 1e-2:
            plt.text(epoch_list[-1], weight,
                     '%s-th_freq-%.1f' % (str(i), freq_list[-1][i]))

    plt.title('weights scatter')

    # fifth subplot: scatter of the freq at each test point, which store on the list freq_list, the x-axies is epoch, y-axis is the freq
    plt.subplot(3, 3, 5)
    for i in range(Q):
        plt.scatter(epoch_list, [f[i] for f in freq_list], s=10)
    plt.title('freq scatter')

    # sixth subplot: scatter of the ls at each test point, which store on the list ls_list, the x-axies is epoch, y-axis is the ls
    plt.subplot(3, 3, 6)
    for i in range(Q):
        plt.scatter(epoch_list, [l[i] for l in ls_list], s=10)

        # if the ls is significant, label each scatter plot with the corresponding id and freq
        ls = [l[i] for l in ls_list][-1]
        if ls > 1e-2:
            plt.text(epoch_list[-1], ls,
                     '%s-th_freq-%.1f' % (str(i), freq_list[-1][i]))
    plt.title('ls scatter')

    # seventh subplot: scatter of the matern_w at each test point, which store on the list matern_w_list, the x-axies is epoch, y-axis is the matern_w
    plt.subplot(3, 3, 7)
    plt.scatter(epoch_list, matern_w_list, s=10)
    plt.title('extra-matern weights scatter')

    plt.subplot(3, 3, 8)
    plt.scatter(epoch_list, matern_ls_list, s=10)
    plt.title('extra-matern ls scatter')

    fix_prefix_dict = {1: '_fix_', 0: '_nonfix_'}
    fix_prefix = 'w'+fix_prefix_dict[model.fix_dict['log-w']]+'ls' + \
        fix_prefix_dict[model.fix_dict['log-ls']] + \
        'freq'+fix_prefix_dict[model.fix_dict['freq']]

    # make the whole figure title to be the name of the trick
    plt.suptitle(
        fix_prefix + '\n' + model.trick_paras['init_u_trick'].__name__ +
        '-nU-%d-Q-%d-epoch-%d-lr-%.4f-freqscale-%d' %
        (num_u_trick, Q, nepoch, model.trick_paras['lr'],
         model.trick_paras['freq_scale']), )

    prefix = 'result_analysis/' + model.trick_paras['equation'] + '/kernel_' + \
        model.cov_func.__class__.__name__ + '-extra-'+model.cov_func_extra.__class__.__name__ + \
        '/epoch_'+str(nepoch)+'/Q'+str(Q)+'/'

    # build the folder if not exist
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    fig_name = 'llk_w-%.1f-' % (model.llk_weight) + model.trick_paras[
        'init_u_trick'].__name__ + '-nU-%d-Q-%d-epoch-%d-lr-%.4f-freqscale=%d-logdet-%d' % (
            num_u_trick, Q, nepoch, model.trick_paras['lr'],
            model.trick_paras['freq_scale'],
            model.trick_paras['logdet']) + other_paras

    print('save fig to ', prefix + fig_name + '.png')

    plt.savefig(prefix + fig_name + '.png')


def make_fig_2d_extra_GP(model, params_extra, log_dict, other_paras=''):
    fontsize = 36
    # plot a figure with 9 subplots. 1 for the truth- prediction, 2 for the loss curve, 3 for the error curve, {4,5,6} for scatter of the weights, freq, and ls for x1, {7,8,9} for scatter of the weights, freq, and ls for x2

    loss_list = log_dict['loss_list']
    err_list = log_dict['err_list']
    epoch_list = log_dict['epoch_list']
    w_list_k1 = log_dict['w_list_k1']
    w_list_k2 = log_dict['w_list_k2']
    freq_list_k1 = log_dict['freq_list_k1']
    freq_list_k2 = log_dict['freq_list_k2']
    ls_list_k1 = log_dict['ls_list_k1']
    ls_list_k2 = log_dict['ls_list_k2']

    matern_w_list_k1 = log_dict['matern_w_list_k1']
    matern_w_list_k2 = log_dict['matern_w_list_k2']

    matern_ls_list_k1 = log_dict['matern_ls_list_k1']
    matern_ls_list_k2 = log_dict['matern_ls_list_k2']

    Q = model.trick_paras['Q']
    nepoch = model.trick_paras['nepoch']
    num_u_trick = model.trick_paras['num_u_trick']

    plt.figure(figsize=(28, 21))

    # first subplot- pred of 2d possion
    plt.subplot(3, 4, 1)
    preds, _ = model.pred_func(params_extra)

    plt.imshow(preds, cmap="hot")
    plt.title('pred-2d:loss = %g, err = %g' % (loss_list[-1], err_list[-1]),
              fontsize=fontsize * 0.5)

    # second subplot: ground truth of 2d possion
    plt.subplot(3, 4, 2)
    plt.imshow(model.ute, cmap="hot")
    plt.title('ground-truth-2d', fontsize=fontsize * 0.5)

    plt.subplot(3, 4, 3)
    plt.plot(epoch_list, loss_list)
    plt.title('loss curve')

    plt.subplot(3, 4, 4)
    plt.plot(epoch_list, err_list)
    plt.title('error curve', fontsize=fontsize * 0.5)

    # fourth subplot: scatter of the weights at each test point, which store on the list w_list, the x-axies is epoch, y-axis is the weights, make the marker size smaller to make the plot clearer

    plt.subplot(3, 4, 5)
    # for i in range(Q):
    #     plt.scatter(epoch_list, [w[i] for w in w_list_k1], s=10)
    for i in range(Q):
        plt.scatter(epoch_list, [w[i] for w in w_list_k1], s=10)

        # if the weight is significant, label each scatter plot with the corresponding id and freq
        weight = [w[i] for w in w_list_k1][-1]
        if weight > 1e-2:
            plt.text(epoch_list[-1], weight,
                     '%s-th_freq-%.1f' % (str(i), freq_list_k1[-1][i]))
    plt.title('weights scatter-k1', fontsize=fontsize * 0.5)

    # fifth subplot: scatter of the freq at each test point, which store on the list freq_list, the x-axies is epoch, y-axis is the freq
    plt.subplot(3, 4, 6)
    for i in range(Q):
        plt.scatter(epoch_list, [f[i] for f in freq_list_k1], s=10)
    plt.title('freq scatter-k1', fontsize=fontsize * 0.5)

    # sixth subplot: scatter of the ls at each test point, which store on the list ls_list, the x-axies is epoch, y-axis is the ls
    plt.subplot(3, 4, 7)

    for i in range(Q):
        plt.scatter(epoch_list, [l[i] for l in ls_list_k1], s=10)

        # if the ls is significant, label each scatter plot with the corresponding id and freq
        ls = [l[i] for l in ls_list_k1][-1]
        if ls > 1e-2:
            plt.text(epoch_list[-1], ls,
                     '%s-th_freq-%.1f' % (str(i), freq_list_k1[-1][i]))

    plt.title('ls scatter-k1', fontsize=fontsize * 0.5)

    plt.subplot(3, 4, 8)
    plt.scatter(epoch_list, matern_w_list_k1, s=10, label='k1')
    plt.scatter(epoch_list, matern_w_list_k2, s=10, label='k2')
    # show legend
    plt.legend()
    plt.title('extra-matern weights scatter')

    plt.subplot(3, 4, 9)
    for i in range(Q):
        plt.scatter(epoch_list, [w[i] for w in w_list_k2], s=10)

        # if the weight is significant, label each scatter plot with the corresponding id and freq
        weight = [w[i] for w in w_list_k2][-1]
        if weight > 1e-2:
            plt.text(epoch_list[-1], weight,
                     '%s-th_freq-%.1f' % (str(i), freq_list_k2[-1][i]))

    plt.title('weights scatter-k2', fontsize=fontsize * 0.5)

    plt.subplot(3, 4, 10)
    for i in range(Q):
        plt.scatter(epoch_list, [f[i] for f in freq_list_k2], s=10)
    plt.title('freq scatter-k2', fontsize=fontsize * 0.5)

    plt.subplot(3, 4, 11)
    for i in range(Q):
        plt.scatter(epoch_list, [l[i] for l in ls_list_k2], s=10)

        # if the ls is significant, label each scatter plot with the corresponding id and freq
        ls = [l[i] for l in ls_list_k2][-1]
        if ls > 1e-2:
            plt.text(epoch_list[-1], ls,
                     '%s-th_freq-%.1f' % (str(i), freq_list_k2[-1][i]))

    plt.title('ls scatter-k2', fontsize=fontsize * 0.5)

    plt.subplot(3, 4, 12)
    plt.scatter(epoch_list, matern_ls_list_k1, s=10, label='k1')
    plt.scatter(epoch_list, matern_ls_list_k2, s=10, label='k2')
    # show legend
    plt.legend()
    plt.title('extra-matern ls scatter')

    fix_prefix_dict = {1: '_fix_', 0: '_nonfix_'}
    fix_prefix = 'w'+fix_prefix_dict[model.fix_dict['log-w']]+'ls' + \
        fix_prefix_dict[model.fix_dict['log-ls']] + \
        'freq'+fix_prefix_dict[model.fix_dict['freq']]

    # make the whole figure title to be the name of the trick
    plt.suptitle(fix_prefix + '\n' +
                 model.trick_paras['init_u_trick'].__name__ +
                 '-nU-%d-Q-%d-epoch-%d-lr-%.4f' %
                 (num_u_trick, Q, nepoch, model.trick_paras['lr']),
                 fontsize=fontsize)

    prefix = 'result_analysis/' + model.trick_paras['equation'] + '/kernel_' + \
        model.cov_func.__class__.__name__ + '-extra-'+model.cov_func_extra.__class__.__name__ + \
        '/epoch_'+str(nepoch)+'/Q'+str(Q)+'/'

    # build the folder if not exist
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    fig_name = 'llk_w-%.1f-' % (model.llk_weight) + model.trick_paras[
        'init_u_trick'].__name__ + '-nu-%d-Q-%d-epoch-%d-lr-%.4f-freqscale=%d-logdet-%d' % (
            num_u_trick, Q, nepoch, model.trick_paras['lr'],
            model.trick_paras['freq_scale'],
            model.trick_paras['logdet']) + other_paras
    print('save fig to ', prefix + fig_name)

    plt.savefig(prefix + fig_name + '.png')


def get_prefix(model, trick_paras):

    if trick_paras['kernel_extra'] is not None:

        prefix = 'result_log/' + trick_paras['equation'] + '/kernel_' + \
            model.cov_func.__class__.__name__ + '-extra-'+model.cov_func_extra.__class__.__name__ + \
            '/epoch_'+str(trick_paras['nepoch']) + \
            '/Q'+str(trick_paras['Q'])+'/'
    else:
        prefix = 'result_log/' + trick_paras['equation'] + '/kernel_' + \
            model.cov_func.__class__.__name__ + \
            '/epoch_'+str(trick_paras['nepoch']) + \
            '/Q'+str(trick_paras['Q'])+'/'

    # build the folder if not exist
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    return prefix


def get_save_name(trick_paras):
    save_name = 'llk_weight-%.1f-nu-%d-Q-%d-epoch-%d-lr-%.4f-freqscale=%d-logdet-%d' % (
        trick_paras['llk_weight'], trick_paras['num_u_trick'],
        trick_paras['Q'], trick_paras['nepoch'], trick_paras['lr'],
        trick_paras['freq_scale'],
        trick_paras['logdet']) + trick_paras['other_paras']
    return save_name


def store_model(model, log_dict, trick_paras):
    prefix = get_prefix(model, trick_paras)
    save_name = get_save_name(trick_paras)

    # save the model, log_dict, and  trick_paras as pickle file
    params = model.params

    if trick_paras['kernel_extra'] is not None:
        params_extra = model.params_extra
        data = (params, params_extra, log_dict, trick_paras)

    else:
        data = (params, log_dict, trick_paras)

    with open(prefix + save_name + '.pkl', 'wb') as f:
        pickle.dump(data, f)

    print('save model, log_dict, trick_paras to ', prefix + save_name + '.pkl')


def wrirte_log(model, err_dict, trick_paras):
    prefix = get_prefix(model, trick_paras)

    # write the log to file, append to the file, build the file if not exist
    with open(prefix + 'log.txt', 'a+') as f:
        f.write(
            'llk_weight-%.1f--nu-%d-Q-%d-epoch-%d-lr-%.4f-freqscale=%d-logdet-%d'
            % (trick_paras['llk_weight'], trick_paras['num_u_trick'],
               trick_paras['Q'], trick_paras['nepoch'], trick_paras['lr'],
               trick_paras['freq_scale'], trick_paras['logdet']) +
            trick_paras['other_paras'] + '\n')

        f.write(
            'err_mean: %.4f, err_std: %.4f, used_time: %.4f, avg_time: %.4f, avg_epochs %d \n'
            % (err_dict['mean'], err_dict['std'], err_dict['used_time'],
               err_dict['avg_time'], err_dict['stop_epoch_mean']))

        f.write('err_list: ' + str(err_dict['err_list']) + '\n\n\n')

    print('write log to ', prefix + 'log.txt')


def get_model_1d(params, trick_paras, new_test = False):

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

    if new_test:
        M = new_test
    else:
        M = 300
    X_test = np.linspace(0, 1, num=M).reshape(-1, 1) * trick_paras['x_scale']
    Y_test = u(X_test)
    # collocation points
    # N_col = 200
    N_col = trick_paras['N_col']

    X_col = np.linspace(0, 1, num=N_col).reshape(-1, 1)* trick_paras['x_scale']
    Xind = np.array([0, X_col.shape[0] - 1])
    y = jnp.array([u(X_col[Xind[0]]), u(X_col[Xind[1]])]).reshape(-1)

    eq_type = trick_paras['equation'].split('-')[0]
    src_vals = model_GP_solver_1d.get_source_val(u, X_col.reshape(-1), eq_type)

    model = model_GP_solver_1d.GP_solver_1d_single(
        Xind,
        y,
        X_col,
        src_vals,
        1e-6,
        X_test,
        Y_test,
        trick_paras,
    )
    model.params = params

    preds, _ = model.preds(params, model.Xte)

    Xtr = model.X_col[model.Xind]

    # plt.plot(model.Xte.flatten(), model.yte.flatten(), 'k-', label='Truth')
    # plt.plot(model.Xte.flatten(), preds.flatten(), 'r-', label='Pred')
    # plt.scatter(Xtr.flatten(), model.y.flatten(), c='g', label='Train')

    return model, preds, Xtr

def get_model_1d_extra(params, params_extra, trick_paras,new_test = False):

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

    if new_test:
        M = new_test
    else:
        M = 300
    X_test = np.linspace(0, 1, num=M).reshape(-1, 1)* trick_paras['x_scale']
    Y_test = u(X_test)
    # collocation points
    N_col = trick_paras['N_col']
    X_col = np.linspace(0, 1, num=N_col).reshape(-1, 1)* trick_paras['x_scale']
    Xind = np.array([0, X_col.shape[0] - 1])
    y = jnp.array([u(X_col[Xind[0]]), u(X_col[Xind[1]])]).reshape(-1)

    eq_type = trick_paras['equation'].split('-')[0]
    src_vals = model_GP_solver_1d.get_source_val(u, X_col.reshape(-1), eq_type)

    model = model_GP_solver_1d_extra.GP_solver_1d_extra(
        Xind,
        y,
        X_col,
        src_vals,
        1e-6,
        X_test,
        Y_test,
        trick_paras,
    )
    model.params = params
    model.params_extra = params_extra

    preds, _ = model.preds_extra(params_extra, model.Xte)
    Xtr = model.X_col[model.Xind]

    return model, preds, Xtr



def get_model_2d(params, trick_paras,new_test = False):

    # equation
    equation_dict = {
        'poisson_2d-sin_cos':
        lambda x, y: jnp.sin(100 * x) * jnp.cos(100 * y),
        'poisson_2d-sin_sin':
        lambda x, y: jnp.sin(100 * x) * jnp.sin(100 * y),
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

    if new_test:
        M = new_test
    else:
        M = 300
    x_pos_test, y_pos_test, u_test_mh = model_GP_solver_2d.get_mesh_data(u, M, M,scale )
    # collocation points  in each dimension
    N = trick_paras['N_col']
    x_pos_tr, y_pos_tr, u_mh = model_GP_solver_2d.get_mesh_data(u, N, N,scale )
    bvals = model_GP_solver_2d.get_boundary_vals(u_mh)

    src_vals = model_GP_solver_2d.get_source_val(u, x_pos_tr, y_pos_tr, eq_type)
    src_vals = src_vals.reshape((x_pos_tr.size, y_pos_tr.size))
    X_test = (x_pos_test, y_pos_test)
    u_test = u_test_mh
    X_col = (x_pos_tr, y_pos_tr)

    model = model_GP_solver_2d.GP_solver_2d_single(
            bvals,
            X_col,
            src_vals,
            1e-6,
            X_test,
            u_test,
            trick_paras,
        )
    model.params = params

    preds,_= model.preds(params)

    # Xtr = model.X_col[model.Xind]

    return model, preds

def get_model_2d_advection(params, trick_paras,new_test = False):

    # equation
    beta = trick_paras['beta']
    equation_dict = {
        'advection-sin':
        lambda x, y: jnp.sin(x-beta*y),
    }

    u = equation_dict[trick_paras['equation']]
    eq_type = trick_paras['equation'].split('-')[0]

    if new_test:
        M = new_test
    else:
        M = 300
    x_pos_test, y_pos_test, u_test_mh = model_GP_solver_2d_advection.get_mesh_data(u, M, M)
    # collocation points  in each dimension
    N = 200
    x_pos_tr, y_pos_tr, u_mh = model_GP_solver_2d_advection.get_mesh_data(u, N, N)
    bvals = model_GP_solver_2d_advection.get_boundary_vals(u_mh)

    src_vals = model_GP_solver_2d_advection.get_source_val(u, x_pos_tr, y_pos_tr, eq_type,beta)
    src_vals = src_vals.reshape((x_pos_tr.size, y_pos_tr.size))
    X_test = (x_pos_test, y_pos_test)
    u_test = u_test_mh
    X_col = (x_pos_tr, y_pos_tr)

    model = model_GP_solver_2d_advection.GP_solver_2d_single_advection(
            bvals,
            X_col,
            src_vals,
            1e-6,
            X_test,
            u_test,
            trick_paras,
        )
    model.params = params

    preds,_= model.preds(params)

    # Xtr = model.X_col[model.Xind]

    return model, preds


class Config(object):
    
    def parse(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        #

        print('=================================')
        print('*', self.config_name)
        print('---------------------------------')

        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print('-', k, ':', getattr(self, k))
        print('=================================')

    def __str__(self, ):

        buff = ""
        buff += '=================================\n'
        buff += ('*' + self.config_name + '\n')
        buff += '---------------------------------\n'

        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                buff += ('-' + str(k) + ':' + str(getattr(self, k)) + '\n')
            #
        #
        buff += '=================================\n'

        return buff

class ExpConfig(Config):

    # equation_list = [
    #     'poisson_2d-sin_cos',
    #     'poisson_2d-sin_cos',
    #     'allencahn_2d-mix-sincos',
    # ]
    #
    # kernel_list = [
    #     Matern52_Cos_1d,
    #     SE_Cos_1d,
    #     Matern52_1d,
    #     SE_1d,
    # ]

    equation = None
    kernel = None
    nepoch = 1000000
    

    def __init__(self, ):
        super(ExpConfig, self).__init__()
        self.config_name = 'Exp Config'
