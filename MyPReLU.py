from keras import backend as K
from keras import constraints, initializers, regularizers
from keras.engine.base_layer import Layer

class MyPReLU(Layer): #PReLUを独自に作成
    def __init__(self,
                alpha_initializer = 'zeros',
                alpha_regularizer = None,
                alpha_constraint = None,
                shared_axes = None,
                **kwargs):
        super(MyPReLU, self).__init__(**kwargs)

        self.alpha_initializer = initializers.get('zeros')
        self.alpha_regularizer = regularizers.get(None)
        self.alpha_constraint = constraints.get(None)

    def build(self, input_shape):
        param_shape = tuple(1 for i in range(len(input_shape) - 1)) + input_shape[-1:]
        self.alpha = self.add_weight(shape = param_shape,
                                     name = 'alpha',
                                     initializer = self.alpha_initializer,
                                     regularizer = self.alpha_regularizer,
                                     constraint = self.alpha_constraint)
        self.built = True

    def call(self, inputs, mask=None):
        pos = K.relu(inputs)
        neg = -self.alpha * K.relu(-inputs)
        return pos + neg

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            }
        base_config = super(MyPReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))