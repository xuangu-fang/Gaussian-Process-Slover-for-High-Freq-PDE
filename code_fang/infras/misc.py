import os, sys
# import pickle
import logging
import numpy as np


# def create_path(path): 
#     try:
#         if not os.path.exists(path):
#             os.makedirs(path, exist_ok=True)
#             print("Directory '%s' created successfully" % (path))
#         #
#     except OSError as error:
#         print("Directory '%s' can not be created" % (path))
#     #

def create_path(path, verbose=True): 
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            if verbose:
                print("Directory '%s' created successfully" % (path))
        #
    except OSError as error:
        print("Directory '%s' can not be created" % (path))
    #
    
# def get_logger(logpath, displaying=True, saving=True, debug=False, append=False):
#     logger = logging.getLogger()
#     if debug:
#         level = logging.DEBUG
#     else:
#         level = logging.INFO
#     logger.setLevel(level)
#     if saving:
#         if append:
#             info_file_handler = logging.FileHandler(logpath, mode="a")
#         else:
#             info_file_handler = logging.FileHandler(logpath, mode="w+")
#         #
#         info_file_handler.setLevel(level)
#         logger.addHandler(info_file_handler)
#     if displaying:
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(level)
#         logger.addHandler(console_handler)
#
#     return logger
    
def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()


class CostsExpDecayScheduler:
    def __init__(
            self,
            init_costs,
            last_step,
            anneal=False,
        ):

        self.espi = 1e-2
        self.last_step = last_step
        self.anneal = anneal
        self.rate = -np.log(self.espi)/self.last_step

        self.init_costs = np.array(init_costs)/np.sum(np.array(init_costs))
        self.term_costs = np.ones(len(init_costs))/len(init_costs)
        self.steps = 0
        self.diff_costs = self.term_costs-self.init_costs
        # cprint('r', self.init_costs)
        # cprint('b', self.term_costs)

    def step(self, ):
        if self.anneal:
            decay_costs = self.diff_costs * (1.0-np.exp(-self.rate*self.steps))
            curr_costs = decay_costs + self.init_costs
            self.steps+=1
            return curr_costs
        else:
            return self.init_costs
    
    
# class PerformMeters(object):
    
#     def __init__(self, save_path, logger=None, test_interval=0):

        
        
#         self.epochs_rmse_tr = []
#         self.epochs_rmse_te = []
#         self.epochs_nrmse_tr = []
#         self.epochs_nrmse_te = []
        
#         self.steps_rmse_tr = []
#         self.steps_rmse_te = []
#         self.steps_nrmse_tr = []
#         self.steps_nrmse_te = []
        
#         self.epochs_preds = []
#         self.epochs_U_list = []
        
#         self.cnt_epochs = 0
#         self.cnt_steps = 0
        
#         self.save_path = save_path
#         self.logger = logger
        
#         self.test_interval = test_interval
        
#     def add_by_epoch(self, rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau=0.0):
        
#         self.epochs_rmse_tr.append(rmse_tr)
#         self.epochs_rmse_te.append(rmse_te)
#         self.epochs_nrmse_tr.append(nrmse_tr)
#         self.epochs_nrmse_te.append(nrmse_te)
        
#         if self.logger is not None:    
#             self.logger.info('=========================================')
#             self.logger.info('                 Epoch{}               '.format(self.cnt_epochs))
#             self.logger.info('=========================================')          
#             self.logger.info('  # rmse_tr={:.6f},  nrmse_tr={:.6f}'.format(rmse_tr, nrmse_tr))
#             self.logger.info('  # rmse_te={:.6f},  nrmse_te={:.6f}'.format(rmse_te, nrmse_te))
#             self.logger.info('  # tau={:.6f}'.format(tau))
#         #
            
#         self.cnt_epochs += 1
        
#     def add_by_step(self, rmse_tr, rmse_te, nrmse_tr, nrmse_te, tau=0.0):
        
#         self.steps_rmse_tr.append(rmse_tr)
#         self.steps_rmse_te.append(rmse_te)
#         self.steps_nrmse_tr.append(nrmse_tr)
#         self.steps_nrmse_te.append(nrmse_te)
        
#         if self.logger is not None:         
#             self.logger.info('---------------  Steps{} ---------------'.format(self.cnt_steps))
#             self.logger.info('  - rmse_tr={:.6f},  nrmse_tr={:.6f}'.format(rmse_tr, nrmse_tr))
#             self.logger.info('  - rmse_te={:.6f},  nrmse_te={:.6f}'.format(rmse_te, nrmse_te))
#             self.logger.info('  - tau={:.6f}'.format(tau))
#         #
            
#         self.cnt_steps += self.test_interval
        
        
#     def add_pred_by_epoch(self, pred):
#         self.epochs_preds.append(pred)
#         self.logger.info('### pred added')
        
#     def add_U_list_by_epoch(self, U_list_torch):
#         U_list_np = []
#         for U_torch in U_list_torch:
#             U_list_np.append(U_torch.data.cpu().numpy())
#         #
#         self.epochs_U_list.append(U_list_np)

#     def save(self,):
        
#         res = {}
        
#         res['epochs_rmse_tr'] = np.array(self.epochs_rmse_tr)
#         res['epochs_rmse_te'] = np.array(self.epochs_rmse_te)
#         res['epochs_nrmse_tr'] = np.array(self.epochs_nrmse_tr)
#         res['epochs_nrmse_te'] = np.array(self.epochs_nrmse_te)
        
#         res['steps_rmse_tr'] = np.array(self.steps_rmse_tr)
#         res['steps_rmse_te'] = np.array(self.steps_rmse_te)
#         res['steps_nrmse_tr'] = np.array(self.steps_nrmse_tr)
#         res['steps_nrmse_te'] = np.array(self.steps_nrmse_te)
        
#         res['epochs_preds'] = np.array(self.epochs_preds)
#         res['epochs_U_list'] = self.epochs_U_list
        

#         with open(os.path.join(self.save_path, 'error.pickle'), 'wb') as handle:
#             pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         #

