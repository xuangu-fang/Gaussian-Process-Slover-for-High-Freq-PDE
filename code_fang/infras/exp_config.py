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
    max_epochs = 1000000

    def __init__(self, ):
        super(ExpConfig, self).__init__()
        self.config_name = 'Exp Config'
