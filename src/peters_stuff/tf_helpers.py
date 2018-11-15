import os
import pickle
import random
import string
from contextlib import contextmanager
from typing import Generic, TypeVar, Optional
import tensorflow.contrib.graph_editor as ge

import tensorflow as tf

from artemis.fileman.local_dir import get_artemis_data_path


def _generate_random_model_path(code_gen_len=16):
    code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(code_gen_len))
    model_path = get_artemis_data_path('tests/models/{}/model'.format(code), make_local_dir=True)
    return model_path


def save_model_and_graph(sess, nodes, model_path = None, code_gen_len=16):
    """

    :param sess:
    :param nodes:
    :param model_path:
    :param code_gen_len:
    :return: The path to the model.  This will
    """
    node_class = nodes.__class__
    if not isinstance(nodes, dict):
        try:
            nodes = nodes._asdict()
        except:
            raise Exception("Nodes must be a dict or have an _asdict method (like a namedtuple).  Don't know how to deal with: {}".format(nodes))


    subgraphdef = tf.graph_util.extract_sub_graph(sess.graph_def, dest_nodes = [tensor.op.name for tensor in nodes.values()])

    temp_graph = tf.Graph()
    with temp_graph.as_default():
        tf.graph_util.import_graph_def(subgraphdef)
        saver = tf.train.Saver()
        saver.save(sess, save_path=model_path, write_meta_graph=True)


    node_names= {name: node.name for name, node in nodes.items()}
    #
    # saver = tf.train.Saver()
    # if model_path is None:
    #     model_path = _generate_random_model_path(code_gen_len=code_gen_len)
    #
    # ancestor_ops = tf.contrib.graph_editor.get_backward_walk_ops(list(nodes.values()), stop_at_ts=[])
    # all_ops_in_graph = [sess.graph.get_operation_by_name(name) for name in [n.name for n in tf.get_default_graph().as_graph_def().node]]
    #
    # names_to_keep = [op.name for op in ancestor_ops] + ['save']
    # # names_to_keep = [op.name for op in ancestor_ops]
    #
    # all_relevant_ops = [op for op in all_ops_in_graph if any(op.name.startswith(name) for name in names_to_keep)]
    #
    # subgraphdef = tf.graph_util.extract_sub_graph(sess.graph_def, dest_nodes = [op.name for op in all_relevant_ops])
    #
    # tf.train.export_meta_graph(filename=model_path +'.meta', graph_def=subgraphdef, saver_def = saver.saver_def)
    # saver.save(sess, save_path=model_path, write_meta_graph=False)
    # saver.save(sess, save_path=model_path, write_meta_graph=True)

    # new_saver = tf.train.import_meta_graph(model_path+'.meta', import_scope='rrrr')
    # new_saver.restore(sess=sess, save_path = model_path)

    model_dir, _ = os.path.split(model_path)

    with open(os.path.join(model_dir, 'nodes.pkl'), 'wb') as f:
        pickle.dump((node_class, node_names), file=f)
    return model_path

#
# def _generate_random_model_dir(code_gen_len=16):
#     code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(code_gen_len))
#     model_dir = get_artemis_data_path('tests/models/{}/'.format(code), make_local_dir=True)
#     return model_dir
#
#
# def save_model_and_graph(sess, nodes, model_dir = None, code_gen_len=16):
#     if model_dir is None:
#         model_dir = _generate_random_model_dir(code_gen_len=code_gen_len)
#
#     node_class = nodes.__class__
#
#     if not isinstance(nodes, dict):
#         try:
#             nodes = nodes._asdict()
#         except:
#             raise Exception("Nodes must be a dict or have an _asdict method (like a namedtuple).  Don't know how to deal with: {}".format(nodes))
#
#
#     output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=[tensor.op.name for tensor in nodes.values()])
#
#     tf.train.write_graph(output_graph_def, model_dir, 'graph.pb', False)
#     return model_dir



def load_model_and_graph(model_path, sess=None, scope=None):
    model_dir, _ = os.path.split(model_path)

    # graph = tf.Graph()
    # with graph.as_default():
    if sess is None:
        sess = tf.Session()
    graph = tf.get_default_graph()

    new_saver = tf.train.import_meta_graph(model_path+'.meta', import_scope=scope)
    # sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)))  # For some odd reason, we have to do that because just restoring
    # sess.run(tf.global_variables_initializer())  # For some odd reason, we have to do this even though we restore them later.
    new_saver.restore(sess=sess, save_path = model_path)
    with open(os.path.join(model_dir, 'nodes.pkl'), 'rb') as f:
        node_class, node_names = pickle.load(f)
    nodes = node_class(**{name: _lookup_refname(graph=graph, refname=refname, scope=scope) for name, refname in node_names.items()})
    return nodes, sess





def _lookup_refname(refname, graph: tf.Graph, scope: Optional[str]=None):
    """
    Lookup a refname in a graph and return the corresponding op or node.
    :param graph: A tensorflow graph
    :param scope: The name scope to look within
    :param refname: The name of the reference
    :return: An Op or a Tensor
    """
    scoped_refname = (scope+'/' if scope is not None else '')+refname
    if ':' in refname:
        return graph.get_tensor_by_name(scoped_refname)
    else:
        return graph.get_operation_by_name(scoped_refname)


T = TypeVar('T')


class TFGraphClass(Generic[T]):
    """
    A class which implements graph calculations.
    """

    __SESSION_VAR_STANDIN = '__SESSION_VAR_STANDIN'
    __NODE_VAR_STANDIN = '__NODE_VAR_STANDIN'

    def __init__(self, nodes: T, sess=None):
        if sess is None:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
        self.sess = sess
        self.nodes = nodes
        self._model_save_path = _generate_random_model_path()

    def __getstate__(self):
        state = self.__dict__.copy()
        save_model_and_graph(sess = self.sess, nodes=self.nodes, model_path=self._model_save_path)
        for k, v in state.items():
            if v is self.sess:
                state[k] = TFGraphClass.__SESSION_VAR_STANDIN
            elif v is self.nodes:
                state[k] = TFGraphClass.__NODE_VAR_STANDIN
        return state

    def __setstate__(self, state):
        model_path = _OVERRIDE_MODEL_PATH if _OVERRIDE_MODEL_PATH is not None else state['_model_save_path']
        nodes, sess = load_model_and_graph(model_path=model_path, scope = _DEFAULT_LOADING_SCOPE)
        for k, v in state.items():
            if v == TFGraphClass.__SESSION_VAR_STANDIN:
                state[k] = sess
            elif v == TFGraphClass.__NODE_VAR_STANDIN:
                state[k] = nodes
        self.__dict__.update(state)

    def dump(self, model_path=None):
        if model_path is None:
            model_path = self._model_save_path
        file_path = model_path + '-TFGraphObject.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        return model_path

    @staticmethod
    def load(model_path, scope = None):
        file_path = model_path + '-TFGraphObject.pkl'
        with hold_override_model_path(model_path):
            with hold_loading_scope(scope):
                with open(file_path, 'rb') as f:
                    obj = pickle.load(f)
        return obj


_DEFAULT_LOADING_SCOPE = None


@contextmanager
def hold_loading_scope(scope=None):
    global _DEFAULT_LOADING_SCOPE
    oldscope = _DEFAULT_LOADING_SCOPE
    _DEFAULT_LOADING_SCOPE = scope
    yield
    _DEFAULT_LOADING_SCOPE = oldscope


_OVERRIDE_MODEL_PATH = None

@contextmanager
def hold_override_model_path(path):
    global _OVERRIDE_MODEL_PATH
    old = _OVERRIDE_MODEL_PATH
    _OVERRIDE_MODEL_PATH = path
    yield
    _OVERRIDE_MODEL_PATH = old


def replicate_subgraph(inputs, outputs, new_inputs=None):
    """
    Define inputs, outputs which cut off a subgraph, then duplicate this subgraph

    Created by jdehesa https://stackoverflow.com/a/53210523/851699
    :param inputs:
    :param outputs:
    :param new_inputs:
    :return:
    """
    single_input = isinstance(inputs, tf.Tensor)
    single_output = isinstance(outputs, tf.Tensor)
    if single_input:
        assert new_inputs is None or isinstance(new_inputs, tf.Tensor)
        inputs = [inputs]
        if new_inputs is not None:
            new_inputs = [new_inputs]
    if single_output:
        outputs = [outputs]

    if new_inputs is None:
        new_inputs = [ge.make_placeholder_from_tensor(x) for x in inputs]
    elif isinstance(new_inputs, dict):
        new_inputs = [new_inputs[x] if x in new_inputs else ge.make_placeholder_from_tensor(x) for x in inputs]
    assert len(new_inputs)==len(inputs)

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
