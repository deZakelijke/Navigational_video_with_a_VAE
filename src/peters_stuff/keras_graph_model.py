from typing import List
from typing import Tuple

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

    single_input = not isinstance(inputs, (list, tuple))
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
            output_values = graph[(inputs, outputs)]([tensor_dict[name] for name in inputs] if _is_multi(inputs) else tensor_dict[inputs])
            if _is_multi(output_values):
                assert _is_multi(outputs) and len(outputs)==len(output_values), 'PUT ERROR HERE'
                for name, out in zip(outputs, output_values):
                    tensor_dict[name] = out
            else:
                tensor_dict[outputs] = output_values
    return tensor_dict


def input_like(np_input):
    return tf.keras.Input(shape=np_input.shape[1:], dtype=np_input.dtype)


class GraphModel(tf.keras.Model):

    def __init__(self, graph, inputs, output_names=None):
        """
        :param Dict(Tuple[Tuple[str], Tuple[str]], Layer] graph:
        :param List[(str, Input)] inputs:
        :param List[str] output_names:
        """
        input_name_layer_pairs, single_input = _parse_inputs(inputs)  # type: List[Tuple[str, InputLayer]]
        node_levels, inputs, outputs = _parse_graph(graph.keys(), inputs = [name for name, _ in input_name_layer_pairs], outputs = output_names)
        tensor_dict = _create_tensor_graph(graph=graph, node_levels=node_levels, input_dict = dict(input_name_layer_pairs))

        super(GraphModel, self).__init__(
            inputs = input_name_layer_pairs[0][1] if single_input else [layer for name, layer in input_name_layer_pairs],
            outputs = tensor_dict[outputs] if not _is_multi(outputs) else [tensor_dict[name] for name in outputs])
        self._is_initialized = False
        self._graph = graph
        self._input_names = list(name for name, _ in input_name_layer_pairs)
        self._output_names = output_names
        self._tensor_dict = tensor_dict
    #
    # def call(self, inputs, training=None, mask=None):
    #     if not self._is_initialized:
    #         inputs = [in]


    def get_subgraph(self, input_names, output_names):
        keys = _extract_subgraph(self._graph.keys(), input_names, output_names)
        subgraph = {k: self._graph[k] for k in keys}
        return GraphModel(
            graph=subgraph,
            inputs = [(input_names, self._tensor_dict[input_names]) if not _is_multi(input_names) else [(inp, self._tensor_dict[inp]) for inp in input_names]],
            output_names = output_names,
            )

    def get_config(self):
        config = super(GraphModel, self).get_config()
        config['graphmodel'] = dict()
        config['graphmodel']['graph'] = [(inp, out, layer.name) for (inp, out), layer in self._graph.items()]
        config['graphmodel']['input_names'] = list(self._input_names)
        config['graphmodel']['output_names'] = list(self._output_names) if self._output_names is not None else None
        return config

    @classmethod
    def from_config(cls, config, custom_objects = None):
        input_tensors, output_tensors, name, created_layers = _model_from_config(config, custom_objects=custom_objects)
        obj = cls.__new__(cls)
        super(GraphModel, obj).__init__(inputs=input_tensors, outputs=output_tensors, name=name)
        obj._graph = {(inp, out): created_layers[name] for inp, out, name in config['graphmodel']['graph']}
        obj._input_names = config['graphmodel']['input_names']
        obj._output_names = config['graphmodel']['output_names']
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

