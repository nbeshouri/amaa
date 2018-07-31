from keras import backend as K
from keras.layers import GRU, GRUCell
from keras.layers.recurrent import _generate_dropout_mask
from keras import regularizers


class EpisodicGRU(GRU):

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 reset_after=False,
                 **kwargs):

        cell = EpisodicGRUCell(units,
                               activation=activation,
                               recurrent_activation=recurrent_activation,
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               recurrent_initializer=recurrent_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer,
                               recurrent_regularizer=recurrent_regularizer,
                               bias_regularizer=bias_regularizer,
                               kernel_constraint=kernel_constraint,
                               recurrent_constraint=recurrent_constraint,
                               bias_constraint=bias_constraint,
                               dropout=dropout,
                               recurrent_dropout=recurrent_dropout,
                               implementation=implementation,
                               reset_after=reset_after)

        # `super(GRU, self)` is correct here because I want to call
        # `RNN`'s `__init__`, not `GRU`'s.
        super(GRU, self).__init__(cell,
                                  return_sequences=return_sequences,
                                  return_state=return_state,
                                  go_backwards=go_backwards,
                                  stateful=stateful,
                                  unroll=unroll,
                                  **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)


class EpisodicGRUCell(GRUCell):

    def build(self, input_shape):
        super().build((None, input_shape[1] - 1))

    def call(self, inputs, states, training=None):

        attention_weight = inputs[:, -1]
        inputs = inputs[:, :-1]

        # The section below was copied from the standard GRU
        # implementation.

        # BEGIN COPIED SECTION

        h_tm1 = states[0]  # previous memory

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=3)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(h_tm1),
                self.recurrent_dropout,
                training=training,
                count=3)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if 0. < self.dropout < 1.:
            inputs_z = inputs * dp_mask[0]
            inputs_r = inputs * dp_mask[1]
            inputs_h = inputs * dp_mask[2]
        else:
            inputs_z = inputs
            inputs_r = inputs
            inputs_h = inputs

        x_z = K.dot(inputs_z, self.kernel_z)
        x_r = K.dot(inputs_r, self.kernel_r)
        x_h = K.dot(inputs_h, self.kernel_h)
        if self.use_bias:
            x_z = K.bias_add(x_z, self.input_bias_z)
            x_r = K.bias_add(x_r, self.input_bias_r)
            x_h = K.bias_add(x_h, self.input_bias_h)

        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z = h_tm1 * rec_dp_mask[0]
            h_tm1_r = h_tm1 * rec_dp_mask[1]
            h_tm1_h = h_tm1 * rec_dp_mask[2]
        else:
            h_tm1_z = h_tm1
            h_tm1_r = h_tm1
            h_tm1_h = h_tm1

        recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel_z)
        recurrent_r = K.dot(h_tm1_r, self.recurrent_kernel_r)
        if self.reset_after and self.use_bias:
            recurrent_z = K.bias_add(recurrent_z, self.recurrent_bias_z)
            recurrent_r = K.bias_add(recurrent_r, self.recurrent_bias_r)

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        # reset gate applied after/before matrix multiplication
        if self.reset_after:
            recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel_h)
            if self.use_bias:
                recurrent_h = K.bias_add(recurrent_h, self.recurrent_bias_h)
            recurrent_h = r * recurrent_h
        else:
            recurrent_h = K.dot(r * h_tm1_h, self.recurrent_kernel_h)

        hh = self.activation(x_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        # END COPIED SECTION

        # Reshape `attention_weight` from `(None,)` to
        # `(None, recur_size)` like `h` so TensorFlow will allow
        # pointwise multiplication.
        attention_weight = K.expand_dims(attention_weight, 1)
        attention_weight = K.repeat_elements(attention_weight, h.shape[1], 1)

        # Use attention weight to interpolate between the current
        # and previous hidden/output. If `attention_weight` is 0
        # the previous state passes through unchanged.
        h = attention_weight * h + (1 - attention_weight) * h_tm1

        return h, [h]
