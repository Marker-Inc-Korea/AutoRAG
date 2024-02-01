import pytest
from autorag.schema.module import Module
from autorag.support import get_support_modules


# Test cases for supported module types
@pytest.mark.parametrize("module_type, expected_module", [
    ('bm25', get_support_modules('bm25')),
    ('vectordb', get_support_modules('vectordb')),
])
def test_module_from_dict_supported(module_type, expected_module):
    module_dict = {
        'module_type': module_type,
        'param1': 'value1',
        'param2': 'value2',
    }
    module = Module.from_dict(module_dict)
    assert module.module_type == module_type
    assert module.module == expected_module
    assert module.module_param == {k: v for k, v in module_dict.items() if k != 'module_type'}


# Test cases for unsupported module types
@pytest.mark.parametrize("module_type", [
    'unsupported_module',
    'another_unsupported_module',
])
def test_module_from_dict_unsupported(module_type):
    module_dict = {'module_type': module_type}
    with pytest.raises(KeyError):
        Module.from_dict(module_dict)
