import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors, _concat
from tensorflow.python.framework import ops

from src.gqn.gqn_draw import GeneratorLSTMCell, InferenceLSTMCell, _InferenceCellInput, GQNLSTMCell
from src.gqn.gqn_params import PARAMS
from src.gqn.gqn_utils import broadcast_pose, eta, eta_g


def query_pos_inference_rnn(representations, target_frames, sequence_size=12, scope="GQN_RNN",
                inference_input_channels = PARAMS.INFERENCE_INPUT_CHANNELS,
                generator_input_channels = PARAMS.GENERATOR_INPUT_CHANNELS,
                lstm_canvas_channels = PARAMS.LSTM_CANVAS_CHANNELS,
                lstm_output_channels = PARAMS.LSTM_OUTPUT_CHANNELS,
                lstm_kernel_size=PARAMS.LSTM_KERNEL_SIZE):
    """
    Creates the computational graph for the DRAW module in inference mode.
    This is the training time setup where the posterior can be inferred from the
    target image.
    """

    dim_r = representations.get_shape().as_list()
    batch = tf.shape(representations)[0]
    height, width = dim_r[1], dim_r[2]

    query_poses = tf.zeros((batch, 2), dtype=tf.float32)

    generator_cell = GeneratorLSTMCell(
        input_shape=[height, width, generator_input_channels],
        output_channels=lstm_output_channels,
        canvas_channels=lstm_canvas_channels,
        kernel_size=lstm_kernel_size,
        name="GeneratorCell")
    inference_cell = InferenceLSTMCell(
        input_shape=[height, width, inference_input_channels],
        output_channels=lstm_output_channels,
        kernel_size=lstm_kernel_size,
        name="InferenceCell")

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as varscope:
        if not tf.executing_eagerly():
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

        query_poses = broadcast_pose(query_poses, height, width)

        inf_state = inference_cell.zero_state(batch, tf.float32)
        gen_state = generator_cell.zero_state(batch, tf.float32)

        # unroll the LSTM cells
        for step in range(sequence_size):
            # TODO(ogroth): currently no variable sharing, remove?
            # generator and inference cell need to have the same variable scope
            # for variable sharing!

            # input into inference RNN
            inf_input = _InferenceCellInput(
                representations, query_poses, target_frames, gen_state.canvas,
                gen_state.lstm.h)
            # update inference cell
            with tf.name_scope("Inference"):
                (inf_output, inf_state) = inference_cell(inf_input, inf_state, "LSTM_inf")
            # estimate statistics and sample state from posterior
            # mu_q, sigma_q, z_q = compute_eta_and_sample_z(inf_state.lstm.h, scope="Sample_eta_q")

        mu_q, sigma_q = eta(inf_state.lstm.h, channels=2)
        mu_q = tf.reduce_mean(mu_q, axis=(1, 2))
        sigma_q = tf.reduce_mean(sigma_q, axis=(1, 2))
        return mu_q, sigma_q


def broadcast_pose_to_map(vector, height, width, n_pose_channels):
  """
  Broadcasts a pose vector to every pixel of a target image.
  """
  # vector = tf.reshape(vector, [-1, 1, 1, PARAMS.POSE_CHANNELS])
  vector = tf.reshape(vector, [-1, 1, 1, n_pose_channels])
  vector = tf.tile(vector, [1, height, width, 1])
  return vector








def convlstm_image_to_position_encoder(image, batch_size, cell_downsample = 4, n_maps=32, sequence_size=12, n_pose_channels=2):
    batch_size_from_img, sy, sx, img_channels = image.get_shape().as_list()
    lstm = GQNLSTMCell(output_channels=n_maps,
                       input_shape=[sy//cell_downsample, sx//cell_downsample, None],
                       )
    (cell_state, hidden_state) = lstm.zero_state(batch_size, tf.float32)
    input_canvas_and_image = tf.layers.conv2d(
        image,
        filters=n_maps, kernel_size=cell_downsample, strides=cell_downsample,
        padding='VALID', use_bias=False,
        name="DownsampleInferenceInputCanvasAndImage")

    # with tf.variable_scope('ggggnnn', reuse=tf.AUTO_REUSE) as varscope:
    for step in range(sequence_size):
        hidden_state = hidden_state + input_canvas_and_image
        _, (cell_state, hidden_state) = lstm(inputs = {'input': input_canvas_and_image}, state = (cell_state, hidden_state))
    mu_q, sigma_q = eta(hidden_state, channels=n_pose_channels)
    mu_q = tf.reduce_mean(mu_q, axis=(1, 2))
    sigma_q = tf.reduce_mean(sigma_q, axis=(1, 2))
    return mu_q, sigma_q


# def convlstm_position_to_image_decoder(query_poses, image_shape, n_maps, canvas_channels, batch_size=None, cell_downsample=4, sequence_size=12, output_kernel_size=5, n_pose_channels = 2, scope="GQN_RNN", output_type = 'normal'):
#     """
#     Creates the computational graph for the DRAW module in generation mode.
#     This is the test time setup where no posterior can be inferred from the
#     target image.
#     """
#     if batch_size is None:
#         batch_size = query_poses.get_shape()[0]  # This will only work if they have fixed shape
#     sy, sx, img_channels = image_shape
#     height, width = sy//cell_downsample, sx//cell_downsample
#     lstm = GQNLSTMCell(output_channels=n_maps, input_shape=[sy//cell_downsample, sx//cell_downsample, None])
#     (cell_state, hidden_state) = lstm.zero_state(batch_size, tf.float32)
#     canvas = tf.zeros((batch_size, sy, sx, canvas_channels), dtype=tf.float32)
#     query_poses = broadcast_pose_to_map(query_poses, height, width, n_pose_channels=n_pose_channels)
#
#     with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as varscope:
#         for step in range(sequence_size):
#             sub_output, (cell_state, hidden_state) = lstm({'inputs': query_poses}, state=(cell_state, hidden_state))
#             canvas = canvas + tf.layers.conv2d_transpose(
#                 sub_output, filters=canvas_channels, kernel_size=cell_downsample, strides=cell_downsample,
#                 name="UpsampleGeneratorOutput")
#
#     if output_type=='normal':
#         mu_target = eta_g(canvas, channels=img_channels, scope="eta_g", kernel_size = output_kernel_size)
#         logvar_target = eta_g(canvas, channels=img_channels, scope="eta_g_var", kernel_size = output_kernel_size)
#         return mu_target, (tf.nn.softplus(logvar_target + .5) + 1e-8)
#     elif output_type=='bernoulli':
#         logit_mu = eta_g(canvas, channels=img_channels, scope="eta_g", kernel_size = output_kernel_size)
#         return logit_mu
#     else:
#
#         raise Exception(output_type)


class ConvLSTM:
    # TODO(stefan): better description here
    """GeneratorLSTMCell wrapper that upscales output with a deconvolution and
       adds it to a side input."""

    def __init__(self,
                 input_shape,
                 output_channels,
                 kernel_size=5,
                 use_bias=True,
                 forget_bias=1.0,
                 hidden_state_name="h",
                 name="GQNCell"):
        """Construct ConvLSTMCell.
        Args:
          conv_ndims: Convolution dimensionality (1, 2 or 3).
          input_shape: Shape of the input as int tuple, excluding the batch size.
          output_channels: int, number of output channels of the conv LSTM.
          kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
          use_bias: (bool) Use bias in convolutions.
          skip_connection: If set to `True`, concatenate the input to the
            output of the conv LSTM. Default: `False`.
          forget_bias: Forget bias.
          initializers: Unused.
          name: Name of the module.
        Raises:
          ValueError: If `skip_connection` is `True` and stride is different from 1
            or if `input_shape` is incompatible with `conv_ndims`.
        """
        if len(input_shape) - 1 != 2:
            raise ValueError("Invalid input_shape {}.".format(input_shape))

        # TODO(stefan,ogroth): do we want to hard-code here the output size of the
        #                      deconvolution to 4?
        self._input_shape = input_shape
        self._output_channels = output_channels
        self._kernel_size = kernel_size
        self._use_bias = use_bias
        self._forget_bias = forget_bias
        self._hidden_state_name = hidden_state_name
        state_size = tf.TensorShape(self._input_shape[:-1] + [output_channels])
        self._output_size = state_size
        self._state_size = tf.contrib.rnn.LSTMStateTuple(state_size, state_size)

    def zero_state(self, batch_size, dtype):
        return tf.zeros(_concat(batch_size, self._output_size), dtype=dtype), tf.zeros(_concat(batch_size, self._output_size), dtype=dtype)

    def __call__(self, inputs, state):

        cell_state, hidden_state = state

        input_contribution = tf.layers.conv2d(inputs, name="input_LSTM_conv", filters=4 * self._output_channels, kernel_size=self._kernel_size, strides=1, padding='SAME', use_bias=self._use_bias, activation=None)
        hidden_contribution = tf.layers.conv2d(hidden_state, name="hidden_LSTM_conv", filters=4 * self._output_channels, kernel_size=self._kernel_size, strides=1, padding='SAME', use_bias=self._use_bias, activation=None)
        new_hidden = input_contribution + hidden_contribution

        gates = tf.split(value=new_hidden, num_or_size_splits=4, axis=-1)
        input_gate, new_input, forget_gate, output_gate = gates

        with tf.name_scope("Forget"):
            new_cell = tf.nn.sigmoid(forget_gate + self._forget_bias) * cell_state

        with tf.name_scope("Update"):
            new_cell += tf.nn.sigmoid(input_gate) * tf.nn.tanh(new_input)

        with tf.name_scope("Output"):
            output = tf.nn.tanh(new_cell) * tf.nn.sigmoid(output_gate)

        new_state = tf.contrib.rnn.LSTMStateTuple(new_cell, output)

        return output, new_state


def convlstm_position_to_image_decoder(query_poses, image_shape, n_maps, canvas_channels, batch_size=None, cell_downsample=4, sequence_size=12, output_kernel_size=5, n_pose_channels = 2, scope="GQN_RNN", output_type = 'normal'):
    """
    Creates the computational graph for the DRAW module in generation mode.
    This is the test time setup where no posterior can be inferred from the
    target image.
    """
    if batch_size is None:
        batch_size = query_poses.get_shape()[0]  # This will only work if they have fixed shape
    sy, sx, img_channels = image_shape
    height, width = sy//cell_downsample, sx//cell_downsample
    lstm = ConvLSTM(output_channels=n_maps, input_shape=[sy//cell_downsample, sx//cell_downsample, None])
    (cell_state, hidden_state) = lstm.zero_state(batch_size, tf.float32)
    canvas = tf.zeros((batch_size, sy, sx, canvas_channels), dtype=tf.float32)
    query_poses = broadcast_pose_to_map(query_poses, height, width, n_pose_channels=n_pose_channels)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as varscope:
        for step in range(sequence_size):
            sub_output, (cell_state, hidden_state) = lstm(inputs=query_poses, state=(cell_state, hidden_state))
            canvas = canvas + tf.layers.conv2d_transpose(
                sub_output, filters=canvas_channels, kernel_size=cell_downsample, strides=cell_downsample,
                name="UpsampleGeneratorOutput")

    if output_type=='normal':
        mu_target = eta_g(canvas, channels=img_channels, scope="eta_g", kernel_size = output_kernel_size)
        logvar_target = eta_g(canvas, channels=img_channels, scope="eta_g_var", kernel_size = output_kernel_size)
        return mu_target, (tf.nn.softplus(logvar_target + .5) + 1e-8)
    elif output_type=='bernoulli':
        logit_mu = eta_g(canvas, channels=img_channels, scope="eta_g", kernel_size = output_kernel_size)
        return logit_mu
    else:

        raise Exception(output_type)
