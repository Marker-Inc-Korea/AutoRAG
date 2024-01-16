from autorag.nodes.retrieval.run import run_retrieval_node
from autorag.schema import Node
from autorag.schema.module import Module


# Test for Node.get_module_node_params method
def test_get_module_node_params():
    modules = [
        Module.from_dict({'module_type': 'bm25', 'key2': ['value1', 'value2'], 'key3': 'value3', 'key4': ['value4', 'value5']}),
    ]
    node = Node(node_type='retrieval', strategy={'strategy_key': 'strategy_value'}, node_params={'param1': 'value1'},
                modules=modules)

    module_node_params = node.get_param_combinations()

    assert isinstance(module_node_params, list)
    assert module_node_params == [{'param1': 'value1', 'key2': 'value1', 'key3': 'value3', 'key4': 'value4'},
                                  {'param1': 'value1', 'key2': 'value1', 'key3': 'value3', 'key4': 'value5'},
                                  {'param1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': 'value4'},
                                  {'param1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': 'value5'}]


# Test for Node.from_dict class method
def test_from_dict():
    node_dict = {
        'node_type': 'retrieval',
        'strategy': {'strategy_key': 'strategy_value'},
        'modules': [{'module_type': 'bm25'}, {'module_type': 'bm25', 'key2': 'value2'}],
        'extra_param': 'extra_value'
    }

    node = Node.from_dict(node_dict)

    assert node.node_type == 'retrieval'
    assert node.run_node == run_retrieval_node
    assert node.strategy == {'strategy_key': 'strategy_value'}
    assert node.node_params == {'extra_param': 'extra_value'}
    assert all(isinstance(mod, Module) for mod in node.modules)
    assert len(node.modules) == 2
    assert node.modules[0].module_param == {}
    assert node.modules[1].module_param == {'key2': 'value2'}
