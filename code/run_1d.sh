    #     kernel-list:
    #    
    #    - "Matern52_Cos_1d"--->GP-HM-Stm,
    
    #    - "SE_Cos_1d"--->GP-HM-GM,
    
    #    - "Matern52_1d"--->GP-Matern,
    
    #    - "SE_1d"--->GP-SE,
    # 

    #     1d-equation list:
    #    
    #  -'poisson_1d-single_sin' --->1d Poission equation with solution:. sin(100x)

    #  -'poisson_1d-x_time_sinx' --->1d Poission equation with solution:. x * sin(200x)

    # -"poisson_1d-sin_cos"--->1d Poission equation with solution: sin(6x)cos(100x)

    # "poisson_1d-mix_sin" --->1d Poission equation with solution:  sin(x) + 0.1*sin(20x) + 0.05*cos(100x)

    # "poisson_1d-x2_add_sinx" --->1d Poission equation with solution:  sin(500x)-2*(x-0.5)^2

    # "allencahn_1d-single_sin" --->1d Allen-Chan equation with solution:  sin(100 * x)    

    #  "allencahn_1d-sin_cos"--->1d Allen-Chan equation with solution: sin(6x)cos(100x)


python model_GP_solver_1d.py -equation='poisson_1d-single_sin' -kernel='Matern52_Cos_1d' -nepoch=100000

python model_GP_solver_1d.py -equation='poisson_1d-x_time_sinx' -kernel='Matern52_Cos_1d' -nepoch=100000

python model_GP_solver_1d.py -equation='poisson_1d-sin_cos' -kernel='Matern52_Cos_1d' -nepoch=100000

python model_GP_solver_1d.py -equation='allencahn_1d-single_sin' -kernel='Matern52_Cos_1d' -nepoch=100000

python model_GP_solver_1d.py -equation='allencahn_1d-sin_cos' -kernel='Matern52_Cos_1d' -nepoch=100000

# poisson_1d-mix_sin and poisson_1d-x2_add_sinx are the most difficult cases to solve and more iterations are needed, we use 1000000 iterations and the "extra_GP" trick to speed up the convergence. The final relative error can be around 1e-3~1e-4. 

# You can reduce the number of iterations or the N_collocation to speed up the training process.   

python model_GP_solver_1d_extra.py -equation='poisson_1d-mix_sin' -kernel='Matern52_Cos_1d' -nepoch=1000000

python model_GP_solver_1d_extra.py -equation='poisson_1d-x2_add_sinx' -kernel='Matern52_Cos_1d' -nepoch=1000000
