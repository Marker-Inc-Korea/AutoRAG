from typing import Optional, Dict

from autorag.support import get_support_modules


def make_generator_callable_param(generator_dict: Optional[Dict]):
	if "generator_module_type" not in generator_dict.keys():
		generator_dict = {
			"generator_module_type": "llama_index_llm",
			"llm": "openai",
			"model": "gpt-4o-mini",
		}
	module_str = generator_dict.pop("generator_module_type")
	module_class = get_support_modules(module_str)
	module_param = generator_dict
	return module_class, module_param
