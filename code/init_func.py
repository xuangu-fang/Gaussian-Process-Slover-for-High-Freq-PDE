# group of initialization functions of u

import numpy as np
import jax.numpy as jnp

def rand(model_class,trick_dict):

    return np.random.randn(model_class.N_con, trick_dict['num_u_trick'])

def randn(model_class,trick_dict):
    
    return np.random.randn(model_class.N_con, trick_dict['num_u_trick'])

def zeros(model_class,trick_dict):
        
    return np.zeros((model_class.N_con, trick_dict['num_u_trick']))

def linear(model_class,trick_dict):

    
    return np.linspace(model_class.y[0], model_class.y[1], model_class.N_con).reshape(-1, 1)



def linear_randn(model_class,trick_dict):
     
     scale = trick_dict['scale'] if 'scale' in trick_dict else 0.2
     
     if trick_dict['num_u_trick'] ==1: 
         
        return np.linspace(model_class.y[0], model_class.y[1], model_class.N_con).reshape(-1, 1) + np.random.randn(model_class.N_con, 1)*scale
     
     else:

        repeat_linear = np.repeat(np.linspace(model_class.y[0], model_class.y[1], model_class.N_con).reshape(-1, 1), trick_dict['num_u_trick'], axis=1) / trick_dict['num_u_trick']

        return repeat_linear + np.random.randn(model_class.N_con, trick_dict['num_u_trick'])*scale


