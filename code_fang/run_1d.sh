python run_exp_2d.py -equation='poisson_2d-sin_sin' -kernel='Matern52_Cos_1d' -max_epochs=1000000

#python run_exp_2d.py -equation='poisson_2d-sin_cos' -kernel='SE_Cos_1d' -max_epochs=1000000
#
#python run_exp_2d.py -equation='poisson_2d-sin_cos' -kernel='Matern52_1d' -max_epochs=1000000
#
#python run_exp_2d.py -equation='poisson_2d-sin_cos' -kernel='SE_1d' -max_epochs=1000000

python run_exp_2d.py -equation='allencahn_2d-mix-sincos' -kernel='Matern52_Cos_1d' -max_epochs=3000000

python run_exp_2d.py -equation='allencahn_2d-mix-sincos' -kernel='SE_Cos_1d' -max_epochs=3000000

python run_exp_2d.py -equation='allencahn_2d-mix-sincos' -kernel='Matern52_1d' -max_epochs=3000000

python run_exp_2d.py -equation='allencahn_2d-mix-sincos' -kernel='SE_1d' -max_epochs=3000000