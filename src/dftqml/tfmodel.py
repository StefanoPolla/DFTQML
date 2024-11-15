import numpy as np
import tensorflow as tf
import json
from os import path


DTYPE = np.float32


class Denormalization(tf.keras.layers.Layer):
    '''
    NOTE: Keras has a recent layer that is supposed to denormalize outputs,
    which is invoked with the `invert=True` keyword argument in the
    `layers.Normalize` layer. Unfortunately, the normalization has been
    implemented wrong (see [https://github.com/keras-team/keras/issues/17047]).
    For this reason, I am implementing the denormalization layer separately.
    Waiting for the bugfix.
    '''

    def __init__(self, mean=None, variance=None):
        super(Denormalization, self).__init__()
        if mean is not None and variance is not None:
            self.mean = tf.constant(mean, DTYPE)
            self.variance = tf.constant(variance, DTYPE)

    def get_config(self):
        return dict(mean=self.mean.numpy(), variance=self.variance.numpy())

    def adapt(self, data):
        ''' TODO test '''
        self.mean = np.mean(data, dtype=DTYPE)
        self.variance = np.var(data, dtype=DTYPE)

    def call(self, inputs):
        return inputs * tf.sqrt(self.variance) + self.mean


class SymmetryExpansion(tf.keras.layers.Layer):
    '''
    replicate symmetry-shifted versions of the density data along the
    penultimate axis of the tensor.

    input: a tensor of size (..., L, features)
    output: a tensor of size (..., 2*L, L, features).
        Each of the (2*L) slices are generated by sliding and reflecting the
        original (L,) sample.

    There are no trainable parameters.
    '''

    def __init__(self, L):
        super(SymmetryExpansion, self).__init__()
        L = self.Li
        shift_reflect_matrix = np.zeros([L, 2 * L, L], dtype=DTYPE)
        for i_exp in range(L):
            for i_in in range(L):
                shift_reflect_matrix[i_in, i_exp, (i_in + i_exp) % L] = 1.
        for i_exp in range(L, 2 * L):
            for i_in in range(L):
                shift_reflect_matrix[i_in, i_exp, (i_exp - i_in) % L] = 1.
        self.shift_reflect_matrix = tf.constant(shift_reflect_matrix,
                                                dtype=DTYPE)

    def get_config(self):
        return dict(L=self.L)

    def call(self, inputs):
        return tf.einsum('...if, ikl -> ...klf',
                         inputs, self.shift_reflect_matrix)


class MeanFeatures(tf.keras.layers.Layer):
    '''
    implements tf.reduce_mean on the specified axis

    input: a tensor of size (..., M, L)
    output: a tensor of size (..., L). Each of the (2*L, L) slices are
        generated by sliding and reflecting the original (L,) sample.

    There are no trainable parameters.
    '''

    def __init__(self, axis):
        super(MeanFeatures, self).__init__()
        self.axis = axis

    def get_config(self):
        return dict(axis=self.axis)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis)


class PeriodicPadding(tf.keras.layers.Layer):
    '''
    Pad the data to maintain translational invariance when combined with a
    Conv1D (choose `padding=valid` for the Conv1D layer)

    input: tensor of shape (..., L, features)
    output: tensor of shape (..., L + kernel_size - 1, features)
    '''

    def __init__(self, kernel_size):
        super(PeriodicPadding, self).__init__()
        self.kernel_size = kernel_size

    def get_config(self):
        return dict(kernel_size=self.kernel_size)

    def call(self, inputs):
        return tf.concat([inputs,
                          tf.gather(inputs,
                                    np.arange(self.kernel_size - 1),
                                    axis=-2)],
                         axis=-2)


def initialize_model(x_training, y_training, *,
                     optimizer=None, enforce_symmetry=False,
                     n_filters=8):

    input_shape = x_training.shape[1:]
    L = input_shape[-1]

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    input_normalization_layer = tf.keras.layers.Normalization(
        mean=np.mean(x_training), variance=np.var(x_training))

    # custom denorm layer, see above.
    # The keras Normalization(invert=True) is bugged.
    output_denormalization_layer = Denormalization(
        mean=np.mean(y_training), variance=np.var(y_training))

    convolutional_layer = tf.keras.layers.Conv1D(filters=n_filters,
                                                 kernel_size=3,
                                                 padding="valid",
                                                 activation='relu')

    if enforce_symmetry:
        flatten_spatial_and_features = tf.keras.layers.Reshape(
            target_shape=list(input_shape[:-1]) + [2 * L, L * n_filters])

        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape),
            input_normalization_layer,
            tf.keras.layers.Reshape([*input_shape, 1]),
            SymmetryExpansion(input_shape[-1]),
            PeriodicPadding(kernel_size=3),
            convolutional_layer,
            flatten_spatial_and_features,
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=1),
            MeanFeatures(axis=-2),
            output_denormalization_layer
        ])

    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape),
            input_normalization_layer,
            tf.keras.layers.Reshape([*input_shape, 1]),
            PeriodicPadding(kernel_size=3),
            convolutional_layer,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=1),
            output_denormalization_layer
        ])

    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    return model


def fit_history(model, x_train, y_train, x_val, y_val,
                *, batch_size, epochs, shuffle=False,
                min_delta=1e-8, patience=5, verbose=1):

    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=min_delta,
                                         patience=patience,
                                         verbose=0)
    ]

    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        verbose=verbose,
                        callbacks=callbacks_list,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=shuffle)

    return history


def save_model(model, model_path, history_dict=None):
    '''alias of model.save. This is for forward-compatibility, if we decide
    more information should be saved'''
    model.save(model_path)
    if history_dict:
        with open(path.join(model_path, 'history.json'), 'wt') as fp:
            json.dump(history_dict, fp)


def load_model(model_path, get_history_dict=False):
    custom_objects = dict(
        Denormalization=Denormalization,
        SymmetryExpansion=SymmetryExpansion,
        MeanFeatures=MeanFeatures,
        PeriodicPadding=PeriodicPadding
    )
    model = tf.keras.models.load_model(model_path,
                                       custom_objects=custom_objects)

    if get_history_dict:
        history_dict = load_history_dict(model_path)
        return model, history_dict
    else:
        return model


def load_history_dict(model_path):
    with open(path.join(model_path, 'history.json'), 'rt') as fp:
        history_dict = json.load(fp)
    return history_dict
