import tensorflow.contrib.graph_editor as ge
import tensorflow as tf


def replicate_subgraph(inputs, outputs, new_inputs=None):

    single_input = isinstance(inputs, tf.Tensor)
    single_output = isinstance(outputs, tf.Tensor)
    if single_input:
        assert new_inputs is None or isinstance(new_inputs, tf.Tensor)
        inputs = [inputs]
        if new_inputs is not None:
            new_inputs = [new_inputs]
    assert len(new_inputs)==len(inputs)
    if single_output:
        outputs = [outputs]

    if new_inputs is None:
        new_inputs = [ge.make_placeholder_from_tensor(x) for x in inputs]

    mappings = {inp: new_inp for inp, new_inp in zip(inputs, new_inputs)}
    # Types of operation that should not be replicated
    # Taken from tensorflow/python/training/device_setter.py
    NON_REPLICABLE = {'Variable', 'VariableV2', 'AutoReloadVariable',
                      'MutableHashTable', 'MutableHashTableV2',
                      'MutableHashTableOfTensors', 'MutableHashTableOfTensorsV2',
                      'MutableDenseHashTable', 'MutableDenseHashTableV2',
                      'VarHandleOp', 'BoostedTreesEnsembleResourceHandleOp'}
    # Find subgraph ops
    ops = tf.contrib.graph_editor.get_backward_walk_ops(outputs, stop_at_ts=mappings.keys())
    # Exclude non-replicable operations
    ops_replicate = [op for op in ops if op.type not in NON_REPLICABLE]
    # Make subgraph viewitems
    sgv = tf.contrib.graph_editor.make_view(*ops_replicate)
    # Make the copy
    _, info = tf.contrib.graph_editor.copy_with_input_replacements(sgv, mappings)
    # Return new outputs
    new_outputs = info.transformed(outputs)
    return (new_inputs[0] if single_input else new_inputs), (new_outputs[0] if single_output else new_outputs)
