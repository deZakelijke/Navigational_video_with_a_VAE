
import tensorflow as tf
import numpy as np
import tensorflow.contrib.graph_editor as ge

def some_function(x):
    w = tf.Variable(initial_value=np.random.randn(4, 5), dtype=tf.float32)
    return tf.tanh(x @ w)

x = tf.placeholder(shape=(None, 4), dtype = tf.float32)
y = some_function(x)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
val_x = np.random.randn(3, 4)
val_y = sess.run(y, feed_dict={x: val_x})


# def get_all_ops(input_tensors, output_tensors):
#
#     ops = set()
#     for o in output_tensors:
#         if o not in input_tensors:
#             ops.add(o.op)
#             ops.update(get_all_ops(input_tensors=input_tensors, output_tensors=o.op.inputs))
#     return ops
#
#
# def copy_subgraph(inputs, outputs, new_inputs = None):
#     sgv = ge.sgv(get_all_ops(inputs, outputs))
#     if new_inputs is None:
#         new_inputs = [ge.make_placeholder_from_tensor(x) for x in inputs]
#     assert len(new_inputs)==len(inputs)
#     new_sgv, _ = ge.copy_with_input_replacements(sgv, replacement_ts={x: x_new for x, x_new in zip(inputs, new_inputs)})
#     new_outputs = new_sgv.outputs
#     return new_inputs, new_outputs
# #
# # x1 = ge.make_placeholder_from_tensor(x)
# # x2 = ge.make_placeholder_from_tensor(x)
# # y1 = ge.copy_with_input_replacements(sgv, replacement_ts={x: x1})
# # y2 = ge.copy_with_input_replacements(sgv, replacement_ts={x: x2})
#
# #
#
# (x1, ), (y1, ) = copy_subgraph(inputs = [x], outputs=[y])
# (x2, ), (y2, ) = copy_subgraph(inputs = [x], outputs=[y])
#
# d = tf.reduce_sum((y1-y2)**2)
#
# val_x1 = np.random.randn(3, 4)
# val_x2 = np.random.randn(3, 4)
#
# sess.run(tf.global_variables_initializer())
# val_d = sess.run([d], feed_dict = {x1: val_x1, x2: val_x2})
#


# Receives the outputs to recalculate and the input replacements
def replicate_subgraph(outputs, mappings):
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
    return info.transformed(outputs)


y2, = replicate_subgraph([y1], {x1: x2})
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(*sess.run([y1, y2], feed_dict={x1: 1, x2: 10}), sep='\n')