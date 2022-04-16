import tensorflow as tf
from tensorflow.keras.layers import Layer, Activation
from tensorflow.keras.initializers import Zeros
from .core import DNN


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
        self._input_check(input_shape)
        self.dnn = DNN(
            self.hidden_units,
            self.activation,
            self.l2_reg,
            self.dropout_rate,
            self.use_bn,
            seed=self.seed,
        )

        super().build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        query, keys = inputs  # * (?, 1, embedding_size), (?, T, embedding_size)
        mask = tf.expand_dims(mask[1], axis=1)  # * (?, 1, T)

        queries = tf.repeat(query, keys.shape[1], 1)  # * (?, T, embedding_size)
        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        # * att_input: (?, T, embedding_size * 4), att_output: (?, T, 1)
        att_out = self.dnn(att_input)
        att_out = tf.transpose(att_out, [0, 2, 1])  # * (?, 1, T)
        att_out = att_out * tf.cast(mask, dtype=tf.float32)
        return att_out

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], 1, input_shape[1][1]

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

    @staticmethod
    def _input_check(input_shape):
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


class AUGRUCell(Layer):
    def __init__(
        self,
        units: int,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        **kwargs
    ):
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation

        self.state_size = [self.units]
        self.output_size = [self.units]

        self.w_u = None
        self.u_u = None
        self.b_u = None
        self.w_r = None
        self.u_r = None
        self.b_r = None
        self.w_h = None
        self.u_h = None
        self.b_h = None
        self.activation_u = Activation(recurrent_activation)
        self.activation_r = Activation(recurrent_activation)
        self.activation_h = Activation(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[0][-1]

        self.w_u, self.u_u, self.b_u = self._gen_weights("u", input_size)
        self.w_r, self.u_r, self.b_r = self._gen_weights("r", input_size)
        self.w_h, self.u_h, self.b_h = self._gen_weights("h", input_size)

        return super().build(input_shape)

    def _gen_weights(self, name, input_size):
        w = self.add_weight(
            "W_" + name,
            shape=(input_size, self.units),
            dtype=tf.float32,
            initializer="glorot_uniform",
            trainable=True,
        )
        u = self.add_weight(
            "U_" + name,
            shape=(self.units, self.units),
            dtype=tf.float32,
            initializer="glorot_uniform",
            trainable=True,
        )
        b = self.add_weight(
            "b_" + name,
            shape=(self.units,),
            dtype=tf.float32,
            initializer=Zeros,
            trainable=True,
        )
        return w, u, b

    def call(self, inputs, hidden, **kwargs):
        x, att_score = inputs
        hidden = hidden[0]

        u = tf.matmul(x, self.w_u) + tf.matmul(hidden, self.u_u) + self.b_u
        u = att_score * self.activation_u(u)

        r = tf.matmul(x, self.w_r) + tf.matmul(hidden, self.u_r) + self.b_r
        r = self.activation_r(r)

        h_hat = tf.matmul(x, self.w_h) + r * tf.matmul(hidden, self.u_h) + self.b_h
        h_hat = self.activation_h(h_hat)

        output = (1 - u) * hidden + u * h_hat
        return output, [output]

    def get_config(self):
        config = {
            "units": self.units,
            "activation": self.activation,
            "recurrent_actvation": self.recurrent_activation,
        }
        base_config = super().get_config()
        return {**config, **base_config}
