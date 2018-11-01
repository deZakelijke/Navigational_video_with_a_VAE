import tensorflow as tf

from src.gqn.gqn_draw import GeneratorLSTMCell, InferenceLSTMCell, _InferenceCellInput
from src.gqn.gqn_params import PARAMS
from src.gqn.gqn_utils import broadcast_pose, compute_eta_and_sample_z, eta


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

    outputs = []
    endpoints = {}
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
