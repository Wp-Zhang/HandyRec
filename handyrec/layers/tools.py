"""Contains some tool layers.

"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Embedding, GRUCell, RNN
from tensorflow.keras.initializers import Zeros
from typing import List
from .layers import DNN


class SequencePoolingLayer(Layer):
    """Pooling layer for sequence feature"""

    def __init__(self, method: str, **kwargs):
        """
        Parameters
        ----------
        method : str
            Pooling method.
        """
        super().__init__(**kwargs)

        assert method in [
            "mean",
            "max",
            "sum",
        ], "Pooling method should be `mean`, `max`, or `sum`"
        self.method = method

    def call(self, inputs, mask=None):
        if mask is None:
            raise ValueError("Embedding layer should set `mask_zero` as True")
        # * inputs: (batch, seq_max_len, emb_dim)
        # * mask: (batch, seq_max_len, emb_dim)
        # * output: (batch, 1, emb_dim)
        mask = tf.dtypes.cast(mask, tf.float32)

        if self.method == "max":
            output = inputs - (1 - mask) * 1e9
            return tf.reduce_max(output, axis=1, keepdims=True)

        elif self.method == "sum":
            output = inputs * mask
            return tf.reduce_sum(output, axis=1, keepdims=True)

        else:  # * self.method == "mean"
            mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True)
            mask_weight = tf.math.divide_no_nan(mask, mask_sum)
            output = inputs * mask_weight
            return tf.reduce_sum(output, axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[-1])

    def get_config(self):
        config = {"method": self.method}
        base_config = super().get_config()
        return {**config, **base_config}


class ValueTable(Layer):
    """Output a full list of values of a feature to be the input of embedding layer"""

    def __init__(self, value_list: List, **kwargs):
        """
        Parameters
        ----------
        value_list : List
            Feature values of all items.
        """
        self.value = tf.constant(value_list)
        super().__init__(**kwargs)

    def call(self, *args, **kwargs):
        return self.value

    def get_config(self):
        return super().get_config()


class SampledSoftmaxLayer(Layer):
    """Sampled softmax"""

    def __init__(self, num_sampled=5, **kwargs):
        """Parameters
        ----------
        num_sampled : int, optional
            Number of sampled negative samples, by default ``5``.
        """
        self.num_sampled = num_sampled
        self.size = None
        self.zero_bias = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.size = input_shape[0][0]
        self.zero_bias = self.add_weight(
            shape=[self.size],
            initializer=Zeros,
            dtype=tf.float32,
            trainable=False,
            name="bias",
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Note
        ----
        `inputs` is a tuple with length of 3.
            inputs[0] is embedding of all items
            inputs[1] is user embedding
            inputs[2] is the index of label items for each user
        """
        item_embeddings, user_embeddings, item_idx = inputs

        loss = tf.nn.sampled_softmax_loss(
            weights=item_embeddings,
            biases=self.zero_bias,
            labels=item_idx,
            inputs=user_embeddings,
            num_sampled=self.num_sampled,
            num_classes=self.size,
        )
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        config = {"num_sampled": self.num_sampled}
        base_config = super().get_config()
        return {**config, **base_config}


class CustomEmbedding(Embedding):
    """
    Rewrite official embedding layer so that masked and un-masked
        embeddings can be concatenated together.
    """

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        # * Rewrite compute_mask
        mask = tf.not_equal(inputs, 0)  # (?, n)
        mask = tf.expand_dims(mask, axis=-1)  # (?, n, 1)
        tile_shape = [1] * (len(mask.shape) - 1) + [self.output_dim]
        mask = tf.tile(mask, tile_shape)  # (?, n, output_dim)
        return mask


class LocalActivationUnit(Layer):
    """The LocalActivationUnit used in DIN"""

    def __init__(
        self,
        hidden_units=(64, 32, 1),
        activation="sigmoid",
        l2_reg=0,
        dropout_rate=0,
        use_bn=False,
        seed=1024,
        **kwargs
    ):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        self.dnn = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.input_check(input_shape)
        self.dnn = DNN(
            self.hidden_units,
            self.activation,
            self.l2_reg,
            self.dropout_rate,
            self.use_bn,
            seed=self.seed,
        )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        query, keys = inputs  # * (?, 1, embedding_size) and (?, T, embedding_size)
        keys_len = keys.get_shape()[1]  # * T
        queries = tf.repeat(query, keys_len, 1)  # * (?, T, embedding_size)
        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        # * att_input: (?, T, embedding_size * 4), att_output: (?, T, 1)
        att_out = self.dnn(att_input)
        att_out = tf.transpose(att_out, [0, 2, 1])  # * (?, 1, T)

        return att_out

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(
        self,
    ):
        config = {
            "activation": self.activation,
            "hidden_units": self.hidden_units,
            "l2_reg": self.l2_reg,
            "dropout_rate": self.dropout_rate,
            "use_bn": self.use_bn,
            "seed": self.seed,
        }
        base_config = super(LocalActivationUnit, self).get_config()
        return {**config, **base_config}

    def input_check(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "A `LocalActivationUnit` layer should be called "
                "on a list of 2 inputs"
            )

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d and %d, expect to be 3 dimensions"
                % (len(input_shape[0]), len(input_shape[1]))
            )

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
            raise ValueError(
                "A `LocalActivationUnit` layer requires "
                "inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)"
                "Got different shapes: %s,%s" % (input_shape[0], input_shape[1])
            )


class AUGRUCell(GRUCell):
    """Implementation of AUGRUCell in DIEN.
    All code is copied from the source code of GRUCell except for one added line.
    """

    def call(self, inputs, states, training=None):
        # * ============================ modyfied ============================
        inputs, att_score = inputs
        # * ============================ modyfied ============================

        h_tm1 = states[0] if tf.nest.is_nested(states) else states  # previous memory

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=3)

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = tf.unstack(self.bias)

        if self.implementation == 1:
            if 0.0 < self.dropout < 1.0:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            x_z = tf.keras.backend.dot(inputs_z, self.kernel[:, : self.units])
            x_r = tf.keras.backend.dot(
                inputs_r, self.kernel[:, self.units : self.units * 2]
            )
            x_h = tf.keras.backend.dot(inputs_h, self.kernel[:, self.units * 2 :])

            if self.use_bias:
                x_z = tf.keras.backend.bias_add(x_z, input_bias[: self.units])
                x_r = tf.keras.backend.bias_add(
                    x_r, input_bias[self.units : self.units * 2]
                )
                x_h = tf.keras.backend.bias_add(x_h, input_bias[self.units * 2 :])

            if 0.0 < self.recurrent_dropout < 1.0:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            recurrent_z = tf.keras.backend.dot(
                h_tm1_z, self.recurrent_kernel[:, : self.units]
            )
            recurrent_r = tf.keras.backend.dot(
                h_tm1_r, self.recurrent_kernel[:, self.units : self.units * 2]
            )
            if self.reset_after and self.use_bias:
                recurrent_z = tf.keras.backend.bias_add(
                    recurrent_z, recurrent_bias[: self.units]
                )
                recurrent_r = tf.keras.backend.bias_add(
                    recurrent_r, recurrent_bias[self.units : self.units * 2]
                )

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = tf.keras.backend.dot(
                    h_tm1_h, self.recurrent_kernel[:, self.units * 2 :]
                )
                if self.use_bias:
                    recurrent_h = tf.keras.backend.bias_add(
                        recurrent_h, recurrent_bias[self.units * 2 :]
                    )
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = tf.keras.backend.dot(
                    r * h_tm1_h, self.recurrent_kernel[:, self.units * 2 :]
                )

            hh = self.activation(x_h + recurrent_h)
        else:
            if 0.0 < self.dropout < 1.0:
                inputs = inputs * dp_mask[0]

            # inputs projected by all gate matrices at once
            matrix_x = tf.keras.backend.dot(inputs, self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = tf.keras.backend.bias_add(matrix_x, input_bias)

            x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = tf.keras.backend.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = tf.keras.backend.bias_add(
                        matrix_inner, recurrent_bias
                    )
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = tf.keras.backend.dot(
                    h_tm1, self.recurrent_kernel[:, : 2 * self.units]
                )

            recurrent_z, recurrent_r, recurrent_h = tf.split(
                matrix_inner, [self.units, self.units, -1], axis=-1
            )

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = tf.keras.backend.dot(
                    r * h_tm1, self.recurrent_kernel[:, 2 * self.units :]
                )

            hh = self.activation(x_h + recurrent_h)
        # previous and candidate state mixed by update gate

        # * ============================ modyfied ============================
        z = z * att_score  # * change this line
        # * ============================ modyfied ============================

        h = z * h_tm1 + (1 - z) * hh
        new_state = [h] if tf.tf.nest.is_tf.nested(states) else h
        return h, new_state


# class AUGRU(RNN):
#     def call(
#         self, inputs, mask=None, training=None, initial_state=None, constants=None
#     ):
#         # The input should be dense, padded with zeros. If a ragged input is fed
#         # into the layer, it is padded and the row lengths are used for masking.
#         inputs, row_lengths = tf.keras.backend.convert_inputs_if_ragged(inputs)
#         is_ragged_input = row_lengths is not None
#         self._validate_args_if_ragged(is_ragged_input, mask)

#         inputs, initial_state, constants = self._process_inputs(
#             inputs, initial_state, constants
#         )

#         self._maybe_reset_cell_dropout_mask(self.cell)
#         if isinstance(self.cell, StackedRNNCells):
#             for cell in self.cell.cells:
#                 self._maybe_reset_cell_dropout_mask(cell)

#         if mask is not None:
#             # Time step masks must be the same for each input.
#             # TODO(scottzhu): Should we accept multiple different masks?
#             mask = tf.nest.flatten(mask)[0]

#         if tf.nest.is_tf.nested(inputs):
#             # In the case of tf.nested input, use the first element for shape check.
#             input_shape = tf.keras.backend.int_shape(tf.nest.flatten(inputs)[0])
#         else:
#             input_shape = tf.keras.backend.int_shape(inputs)
#         timesteps = input_shape[0] if self.time_major else input_shape[1]
#         if self.unroll and timesteps is None:
#             raise ValueError(
#                 "Cannot unroll a RNN if the "
#                 "time dimension is undefined. \n"
#                 "- If using a Sequential model, "
#                 "specify the time dimension by passing "
#                 "an `input_shape` or `batch_input_shape` "
#                 "argument to your first layer. If your "
#                 "first layer is an Embedding, you can "
#                 "also use the `input_length` argument.\n"
#                 "- If using the functional API, specify "
#                 "the time dimension by passing a `shape` "
#                 "or `batch_shape` argument to your Input layer."
#             )

#         kwargs = {}
#         if generic_utils.has_arg(self.cell.call, "training"):
#             kwargs["training"] = training

#         # TF RNN cells expect single tensor as state instead of list wrapped tensor.
#         is_tf_rnn_cell = getattr(self.cell, "_is_tf_rnn_cell", None) is not None
#         # Use the __call__ function for callable objects, eg layers, so that it
#         # will have the proper name scopes for the ops, etc.
#         cell_call_fn = self.cell.__call__ if callable(self.cell) else self.cell.call
#         if constants:
#             if not generic_utils.has_arg(self.cell.call, "constants"):
#                 raise ValueError("RNN cell does not support constants")

#             def step(inputs, states):
#                 constants = states[
#                     -self._num_constants :
#                 ]  # pylint: disable=invalid-unary-operand-type
#                 states = states[
#                     : -self._num_constants
#                 ]  # pylint: disable=invalid-unary-operand-type

#                 states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
#                 output, new_states = cell_call_fn(
#                     inputs, states, constants=constants, **kwargs
#                 )
#                 if not tf.nest.is_tf.nested(new_states):
#                     new_states = [new_states]
#                 return output, new_states

#         else:

#             def step(inputs, states):
#                 states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
#                 output, new_states = cell_call_fn(inputs, states, **kwargs)
#                 if not tf.nest.is_tf.nested(new_states):
#                     new_states = [new_states]
#                 return output, new_states

#         last_output, outputs, states = tf.keras.backend.rnn(
#             step,
#             inputs,
#             initial_state,
#             constants=constants,
#             go_backwards=self.go_backwards,
#             mask=mask,
#             unroll=self.unroll,
#             input_length=row_lengths if row_lengths is not None else timesteps,
#             time_major=self.time_major,
#             zero_output_for_mask=self.zero_output_for_mask,
#         )

#         if self.stateful:
#             updates = [
#                 state_ops.assign(self_state, state)
#                 for self_state, state in zip(
#                     tf.nest.flatten(self.states), tf.nest.flatten(states)
#                 )
#             ]
#             self.add_update(updates)

#         if self.return_sequences:
#             output = tf.keras.backend.maybe_convert_to_ragged(
#                 is_ragged_input, outputs, row_lengths, go_backwards=self.go_backwards
#             )
#         else:
#             output = last_output

#         if self.return_state:
#             if not isinstance(states, (list, tuple)):
#                 states = [states]
#             else:
#                 states = list(states)
#             return generic_utils.to_list(output) + states
#         else:
#             return output


# class Similarity(Layer):
#     def __init__(self, type: str, **kwargs):
#         self.type = type
#         super().__init__(**kwargs)

#     def call(self, inputs, *args, **kwargs):
#         embd_a, embd_b = inputs
#         if self.type == "cos":
#             embd_a = tf.nn.l2_normalize(embd_a, axis=-1)
#             embd_b = tf.nn.l2_normalize(embd_b, axis=-1)
#         output = tf.reduce_sum(tf.multiply(embd_a, embd_b), axis=-1, keepdims=True)
#         return output

#     def get_config(self):
#         config = {"type": self.type}
#         base_config = super().get_config()
#         return {**config, **base_config}
