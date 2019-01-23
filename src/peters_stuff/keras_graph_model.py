from typing import List
from typing import Tuple

import itertools
import tensorflow as tf
from tensorflow.python.keras.utils import get_custom_objects


def _is_multi(tensor_spec):
    return isinstance(tensor_spec, (list, tuple))


def _to_multi(tensor_spec):
    return tuple(tensor_spec) if _is_multi(tensor_spec) else (tensor_spec, )


def _order_node_executions(graph_keys, inputs, outputs):

    resolved_edges = set(inputs)
    node_levels = []
    graph_key_set = set(graph_keys)
    output_set = set(outputs)
    while not output_set.issubset(resolved_edges):  # i.e. while all outputs have not yet been computed
        ready_nodes = [node for node in graph_key_set for inputs, outputs in [node] if set(_to_multi(inputs)).issubset(resolved_edges)]
        graph_key_set = graph_key_set.difference(ready_nodes)
        assert len(ready_nodes)>0, "This should be impossible.  Call Peter."
        node_levels.append(ready_nodes)
        resolved_edges.update({out for node in ready_nodes for inputs, outputs in [node] for out in _to_multi(outputs)})
    return node_levels


def _parse_graph(graph_keys, inputs, outputs, allow_subgraph = False):

    single_input = not isinstance(inputs, (list, tuple)) and not inputs is None
    single_output = not isinstance(outputs, (list, tuple)) and not outputs is None

    if single_input:
        inputs = (inputs, )
    if single_output:
        outputs = (outputs, )

    all_inputs = set(nodename for inputs, _ in graph_keys for nodename in _to_multi(inputs))
    all_outputs = list(nodename for _, outputs in graph_keys for nodename in _to_multi(outputs))
    for nodename in all_outputs:
        assert sum(name==nodename for name in all_outputs)==1, 'Output node "{}" was defined {} times (it should only be defined once)'.format(nodename, sum(name==nodename for name in all_outputs))

    # Check inputs
    inferred_inputs = all_inputs.difference(all_outputs)
    if inputs is None:
        assert len(inferred_inputs)==1, "If you do not specify inputs, there must be only one input in the graph.  Inferred inputs: {}".format(inferred_inputs)
        inputs = tuple(inferred_inputs)
    else:
        if not allow_subgraph:
            assert set(inputs)==inferred_inputs, "The set of inferred input nodes: {}, did not match the set of given input names: {}".format(inferred_inputs, set(inputs))

    # Check outputs
    inferred_outputs = set(all_outputs).difference(all_inputs)
    if outputs is None:
        assert len(inferred_outputs)==1, "If you do not specify outputs, there must be only one output from the graph.  Inferred outputs: {}".format(inferred_outputs)
        outputs = tuple(inferred_outputs)
    else:
        if not allow_subgraph:
            assert set(outputs)==inferred_outputs, "The set of inferred output nodes: {}, did not match the set of given outputs: {}".format(inferred_outputs, set(outputs))

    node_levels = _order_node_executions(graph_keys, inputs = set(inferred_inputs), outputs=_to_multi(outputs))
    # resolved_edges = set(inferred_inputs)
    # node_levels = []
    # graph_key_set = set(graph_keys)
    # while sum(len(lev) for lev in node_levels) < len(graph_keys):
    #     ready_nodes = [node for node in graph_key_set for inputs, outputs in [node] if set(_to_multi(inputs)).issubset(resolved_edges)]
    #     graph_key_set = graph_key_set.difference(ready_nodes)
    #     assert len(ready_nodes)>0, "This should be impossible.  Call Peter."
    #     node_levels.append(ready_nodes)
    #     resolved_edges.update({out for node in ready_nodes for inputs, outputs in [node] for out in _to_multi(outputs)})

    inputs = inputs[0] if single_input else inputs
    outputs = outputs[0] if single_output else outputs
    return node_levels, inputs, outputs


def _extract_subgraph(graph_keys, inputs, outputs):
    inputs = _to_multi(inputs)
    outputs = _to_multi(outputs)

    required_edges = set(outputs)
    # available_edges = set(inputs)

    subgraph = set()
    while len(required_edges)>0:
        subgraph = subgraph.union({(inps, outs) for (inps, outs) in graph_keys if len(set(_to_multi(outs)).intersection(required_edges))>0})
        available_edges = set(inputs).union(set(out for _, outs in subgraph for out in _to_multi(outs)))
        required_edges = set(inp for inps, outs in subgraph for inp in _to_multi(inps)).difference(available_edges)
    return subgraph


def _parse_inputs(inputs):

    single_input = not _is_multi(inputs) or (_is_multi(inputs and len(inputs)==2 and isinstance(inputs[0], str) and isinstance(inputs[1], tf.Tensor)))
    if single_input:
        inputs = (inputs, )

    input_name_layer_pairs = []

    for inp in inputs:
        if isinstance(inp, tuple) and len(inp)==2:
            input_name_layer_pairs.append(inp)
        elif isinstance(inp, tf.Tensor):
            assert inp.op.type == 'Placeholder', "If you pass in a tensor input, it must be a placeholder.  Otherwise use the inputs=[(name, tensor), ...] syntax"
            name = inp.name[:inp.name.index(':')]
            input_name_layer_pairs.append((name, inp))
        else:
            raise Exception('Each input must either be an InputLayer or a (name, Tensor) pair.  Got {}'.format(inp))
    return input_name_layer_pairs, single_input


def _create_tensor_graph(graph, node_levels, input_dict):

    tensor_dict = input_dict.copy()
    for level in node_levels:
        for inputs, outputs in level:
            layer = graph[(inputs, outputs)]
            #
            # if _is_multi(layer):
            #     output_values = [tensor_dict[name] for name in inputs] if _is_multi(inputs) else tensor_dict[inputs]
            #     for lay in layer:
            #         output_values = lay(output_values)
            # else:
            input_tensors = [tensor_dict[name] for name in inputs] if _is_multi(inputs) else tensor_dict[inputs]
            try:
                output_values = layer(input_tensors)
            except:
                print('Error while Processing Node: {}: {} with inputs {}'.format((inputs, outputs), layer, input_tensors))
                raise

            if _is_multi(output_values):
                assert _is_multi(outputs) and len(outputs)==len(output_values), 'PUT ERROR HERE'
                for name, out in zip(outputs, output_values):
                    tensor_dict[name] = out
            else:
                tensor_dict[outputs] = output_values
    return tensor_dict


def input_like(np_input):
    return tf.keras.Input(shape=np_input.shape[1:], dtype=np_input.dtype)


def _to_layer(layer_spec):

    if isinstance(layer_spec, tf.keras.layers.Layer):
        return layer_spec
    elif callable(layer_spec):
        return tf.keras.layers.Lambda(lambda args: layer_spec(*args) if _is_multi(args) else layer_spec(args))
    else:
        raise NotImplementedError(layer_spec)


def _normalize_graph(graph):
    all_nodes = set(name for inp, out in graph.keys() for name in _to_multi(inp)+_to_multi(out))
    anon_node_name_gen = ('anon_{}'.format(i) for i in itertools.count() if i not in all_nodes)
    newgraph = {}
    for (inp, out), layer in graph.items():
        if not _is_multi(layer):
            newgraph[inp, out] = _to_layer(layer)
        else:
            names = [inp] + [next(anon_node_name_gen) for _ in range(len(layer)-1)] + [out]
            for inp_, out_, lay in zip(names[:-1], names[1:], layer):
                newgraph[inp_, out_] = _to_layer(lay)
    return newgraph


class GraphModel(tf.keras.Model):

    def __init__(self, graph, input_names = None, output_names=None, input_tensors=None, inputs_like=None):
        """
        :param Dict(Tuple[Tuple[str], Tuple[str]], Layer] graph:
        :param Union[str, Sequence[str]] input_names:
        :param Union[str, Sequence[str]] output_names:
        """

        self._setattr_tracking = False
        self._is_initialized = False

        # self._layer_graph = {(inp, out: tf.keras.models.Sequential(v) if _is_multi(v) else v for k, v in graph.items()}
        self._layer_graph = _normalize_graph(graph)
        self._node_levels, self._input_names, self._output_names = _parse_graph(self._layer_graph.keys(), inputs = input_names, outputs = output_names)
        if input_tensors is not None:
            self._initialize(input_tensors)
        elif inputs_like is not None:
            self._initialize_from_example_inputs(inputs_like)
        self._setattr_tracking = True

    def _initialize_from_example_inputs(self, input_arrays):
        input_tensors = [input_like(xi) for xi in input_arrays] if _is_multi(input_arrays) else input_like(input_arrays)
        self._initialize(input_tensors)

    def _initialize(self, input_tensors):

        if _is_multi(self._input_names):
            input_tensors = _to_multi(input_tensors)
            assert len(input_tensors)==len(self._input_names), "Graph requires {} input signals {}, but got {} input tensors {}".format(len(self._input_names), self._input_names, len(input_tensors), input_tensors)
            input_dict = {name: tens for name, tens in zip(self._input_names, input_tensors)}
        else:
            if _is_multi(input_tensors):
                assert len(input_tensors)==1, 'Graph only accepts 1 input tensor: {}.  You provided {}: {}'.format(self._input_names, len(input_tensors), input_tensors)
                input_tensors, = input_tensors
            input_dict = {self._input_names: input_tensors}

        tensor_dict = _create_tensor_graph(graph=self._layer_graph, node_levels=self._node_levels, input_dict = input_dict)

        super(GraphModel, self).__init__(
            inputs = input_tensors,
            outputs = tensor_dict[self._output_names] if not _is_multi(self._output_names) else [tensor_dict[name] for name in self._output_names]
        )
        self._is_initialized=True

    def predict(self, x, **kwargs):
        if not self._is_initialized:
            self._initialize_from_example_inputs(x)
        return super(GraphModel, self).predict(x=x, **kwargs)

    def call(self, x, training=None, mask=None):
        if not self._is_initialized:
            self._initialize(x)
        return super(GraphModel, self).call(x, training=training, mask=mask)

    def get_subgraph(self, input_names, output_names):
        keys = _extract_subgraph(self._layer_graph.keys(), input_names, output_names)
        subgraph = {k: self._layer_graph[k] for k in keys}
        return GraphModel(
            graph=subgraph,
            input_names = input_names,
            output_names = output_names,
            # input_tensors = self.input if self._is_initialized else None
            )

    def get_config(self):
        if not self._is_initialized:
            raise NotImplementedError('Cannot yet save uninitialized models.  Either provide input tensors or call the model before saving.')
        config = super(GraphModel, self).get_config()
        config['graphmodel'] = dict()
        config['graphmodel']['graph'] = [(inp, out, layer.name) for (inp, out), layer in self._layer_graph.items()]
        config['graphmodel']['input_names'] = list(self._input_names)
        config['graphmodel']['is_initialized'] = self._is_initialized
        config['graphmodel']['output_names'] = list(self._output_names) if self._output_names is not None else None
        config['graphmodel']['node_levels'] = self._node_levels
        config['graphmodel']['version'] = 0
        return config

    @classmethod
    def from_config(cls, config, custom_objects = None):
        input_tensors, output_tensors, name, created_layers = _model_from_config(config, custom_objects=custom_objects)
        obj = cls.__new__(cls)
        super(GraphModel, obj).__init__(inputs=input_tensors, outputs=output_tensors, name=name)
        obj._layer_graph = {(tuple(inp) if _is_multi(inp) else inp, tuple(out) if _is_multi(out) else out): created_layers[name] for inp, out, name in config['graphmodel']['graph']}
        obj._input_names = config['graphmodel']['input_names']
        obj._output_names = config['graphmodel']['output_names']
        obj._is_initialized = config['graphmodel']['is_initialized']
        obj._node_levels = config['graphmodel']['node_levels']
        return obj


objs = get_custom_objects()
objs['GraphModel'] = GraphModel



def _model_from_config(config, custom_objects=None):
    """
    <Copied and very slightly modified from Keras's Model.from_config()>

    Instantiates a Model from its config (output of `get_config()`).

    Arguments:
        config: Model config dictionary.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A model instance.

    Raises:
        ValueError: In case of improperly formatted config dict.
    """
    # Layer instances created during
    # the graph reconstruction process
    created_layers = {}

    # Dictionary mapping layer instances to
    # node data that specifies a layer call.
    # It acts as a queue that maintains any unprocessed
    # layer call until it becomes possible to process it
    # (i.e. until the input tensors to the call all exist).
    unprocessed_nodes = {}

    def add_unprocessed_node(layer, node_data):
      if layer not in unprocessed_nodes:
        unprocessed_nodes[layer] = [node_data]
      else:
        unprocessed_nodes[layer].append(node_data)

    def process_node(layer, node_data):
      """Deserialize a node.

      Arguments:
          layer: layer instance.
          node_data: node config dict.

      Raises:
          ValueError: In case of improperly formatted `node_data` dict.
      """
      input_tensors = []
      for input_data in node_data:
        inbound_layer_name = input_data[0]
        inbound_node_index = input_data[1]
        inbound_tensor_index = input_data[2]
        if len(input_data) == 3:
          kwargs = {}
        elif len(input_data) == 4:
          kwargs = input_data[3]
        else:
          raise ValueError('Improperly formatted model config.')
        if inbound_layer_name not in created_layers:
          add_unprocessed_node(layer, node_data)
          return
        inbound_layer = created_layers[inbound_layer_name]
        if len(inbound_layer._inbound_nodes) <= inbound_node_index:
          add_unprocessed_node(layer, node_data)
          return
        inbound_node = inbound_layer._inbound_nodes[inbound_node_index]
        input_tensors.append(inbound_node.output_tensors[inbound_tensor_index])
      # Call layer on its inputs, thus creating the node
      # and building the layer if needed.
      if input_tensors:
        if len(input_tensors) == 1:
          layer(input_tensors[0], **kwargs)
        else:
          layer(input_tensors, **kwargs)

    def process_layer(layer_data):
      """Deserializes a layer, then call it on appropriate inputs.

      Arguments:
          layer_data: layer config dict.

      Raises:
          ValueError: In case of improperly formatted `layer_data` dict.
      """
      layer_name = layer_data['name']

      # Instantiate layer.
      from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top

      layer = deserialize_layer(layer_data, custom_objects=custom_objects)
      created_layers[layer_name] = layer

      # Gather layer inputs.
      inbound_nodes_data = layer_data['inbound_nodes']
      for node_data in inbound_nodes_data:
        # We don't process nodes (i.e. make layer calls)
        # on the fly because the inbound node may not yet exist,
        # in case of layer shared at different topological depths
        # (e.g. a model such as A(B(A(B(x)))))
        add_unprocessed_node(layer, node_data)

    # First, we create all layers and enqueue nodes to be processed
    for layer_data in config['layers']:
      process_layer(layer_data)
    # Then we process nodes in order of layer depth.
    # Nodes that cannot yet be processed (if the inbound node
    # does not yet exist) are re-enqueued, and the process
    # is repeated until all nodes are processed.
    while unprocessed_nodes:
      for layer_data in config['layers']:
        layer = created_layers[layer_data['name']]
        if layer in unprocessed_nodes:
          for node_data in unprocessed_nodes.pop(layer):
            process_node(layer, node_data)

    name = config.get('name')
    input_tensors = []
    output_tensors = []
    for layer_data in config['input_layers']:
      layer_name, node_index, tensor_index = layer_data
      assert layer_name in created_layers
      layer = created_layers[layer_name]
      layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
      input_tensors.append(layer_output_tensors[tensor_index])
    for layer_data in config['output_layers']:
      layer_name, node_index, tensor_index = layer_data
      assert layer_name in created_layers
      layer = created_layers[layer_name]
      layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
      output_tensors.append(layer_output_tensors[tensor_index])
    return input_tensors, output_tensors, name, created_layers
    # return cls(inputs=input_tensors, outputs=output_tensors, name=name)

