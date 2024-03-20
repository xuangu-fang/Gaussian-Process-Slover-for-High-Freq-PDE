    #     kernel-list:
    #    
    #    - Matern52_Cos_1d--->GP-HM-Stm,

    #    - SE_Cos_1d--->GP-HM-GM,
    
    #    - Matern52_1d--->GP-Matern,
    
    #    - SE_1d--->GP-SE,
    # 

    #     2d-equation list:
    #    
    #    - poisson_2d-sin_sin --->2d Poission equation with solution:. sin(100 * x) * sin(100 * y) 

    #    - poisson_2d-sin_add_cos --->2d Poission equation with solution:. sin(6 * x) * jnp.cos(20 * x) + jnp.sin(6 * y) * jnp.cos(20 * y)

    #    - allencahn_2d-mix-sincos --->2d Allen-Cahn equation with solution:.(jnp.sin(x) + 0.1 * jnp.sin(20 * x) + jnp.cos(100 * x)) * (jnp.sin(y) + 0.1 * jnp.sin(20 * y) + jnp.cos(100 * y))

    #    - advection-sin--->2d(1d) advection equation with solution:.sin(x-beta*y),


python model_GP_solver_2d.py -equation='poisson_2d-sin_sin' -kernel='Matern52_Cos_1d' -nepoch=100000

python model_GP_solver_2d.py -equation='poisson_2d-sin_add_cos' -kernel='Matern52_Cos_1d' -nepoch=1000000

python model_GP_solver_2d.py -equation='allencahn_2d-mix-sincos' -kernel='Matern52_Cos_1d' -nepoch=3000000

python model_GP_solver_advection.py -equation='advection-sin' -kernel='Matern52_Cos_1d' -nepoch=1000000

