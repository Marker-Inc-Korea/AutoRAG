import os.path
import random
from typing import Optional, List, Dict, Any

import pandas as pd
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM

from autorag.utils.util import process_batch, get_event_loop

package_dir = os.path.dirname(os.path.realpath(__file__))


def generate_qa_llama_index(
	llm: LLM,
	contents: List[str],
	prompt: Optional[str] = None,
	question_num_per_content: int = 1,
	max_retries: int = 3,
	batch: int = 4,
) -> List[List[Dict]]:
	"""
	Generate a qa set from the list of contents.
	It uses a single prompt for all contents.
	If you want to use more than one prompt for generating qa,
	you can consider using generate_qa_llama_index_by_ratio.

	:param llm: Llama index model
	:param contents: List of content strings.
	:param prompt: The prompt to use for the qa generation.
	    The prompt must include the following placeholders:
	    - {{text}}: The content string
	    - {{num_questions}}: The number of questions to generate
	    As default, the prompt is set to the default prompt for the question type.
	:param question_num_per_content: Number of questions to generate for each content.
	    Default is 1.
	:param max_retries: The maximum number of retries when generated question number is not equal to the target number.
	    Default is 3.
	:param batch: The batch size to process asynchronously.
	    Default is 4.
	:return: 2-d list of dictionaries containing the query and generation_gt.
	"""
	# load default prompt
	if prompt is None:
		prompt = open(
			os.path.join(package_dir, "llama_index_default_prompt.txt"), "r"
		).read()

	tasks = [
		async_qa_gen_llama_index(
			content, llm, prompt, question_num_per_content, max_retries
		)
		for content in contents
	]
	loops = get_event_loop()
	results = loops.run_until_complete(process_batch(tasks, batch))
	return results


def generate_answers(
	llm: LLM,
	contents: List[str],
	queries: List[str],
	batch: int = 4,
) -> List[List[Dict]]:
	"""
	Generate qa sets from the list of contents using existing queries.

	:param llm: Llama index model
	:param contents: List of content strings.
	:param queries: List of existing queries.
	:param batch: The batch size to process asynchronously.
	:return: 2-d list of dictionaries containing the query and generation_gt.
	"""

	tasks = [
		generate_basic_answer(llm, content, query)
		for content, query in zip(contents, queries)
	]
	loops = get_event_loop()
	results = loops.run_until_complete(process_batch(tasks, batch))
	return results


def generate_qa_llama_index_by_ratio(
	llm: LLM,
	contents: List[str],
	prompts_ratio: Dict,
	question_num_per_content: int = 1,
	max_retries: int = 3,
	random_state: int = 42,
	batch: int = 4,
) -> List[List[Dict]]:
	"""
	Generate a qa set from the list of contents.
	You can set the ratio of prompts that you want to use for generating qa.
	It distributes the number of questions to generate for each content by the ratio randomly.

	:param llm: Llama index model
	:param contents: List of content strings.
	:param prompts_ratio: Dictionary of prompt paths and their ratios.
	    Example: {"prompt/prompt1.txt": 0.5, "prompt/prompt2.txt": 0.5}
	    The value sum doesn't have to be 1.
	    The path must be the absolute path, and the file must exist.
	    Plus, it has to be a text file which contains proper prompt.
	    Each prompt must contain the following placeholders:
	    - {{text}}: The content string
	    - {{num_questions}}: The number of questions to generate
	:param question_num_per_content: Number of questions to generate for each content.
	    Default is 1.
	:param max_retries: The maximum number of retries when generated question number is not equal to the target number.
	    Default is 3.
	:param random_state: Random seed
	    Default is 42.
	:param batch: The batch size to process asynchronously.
	    Default is 4.
	:return: 2-d list of dictionaries containing the query and generation_gt.
	"""
	prompts = list(map(lambda path: open(path, "r").read(), prompts_ratio.keys()))
	assert all([validate_llama_index_prompt(prompt) for prompt in prompts])

	content_indices = list(range(len(contents)))
	random.seed(random_state)
	random.shuffle(content_indices)

	slice_content_indices: List[List[str]] = distribute_list_by_ratio(
		content_indices, list(prompts_ratio.values())
	)
	temp_df = pd.DataFrame({"idx": slice_content_indices, "prompt": prompts})
	temp_df = temp_df.explode("idx", ignore_index=True)
	temp_df = temp_df.sort_values(by="idx", ascending=True)

	final_df = pd.DataFrame({"content": contents, "prompt": temp_df["prompt"].tolist()})

	tasks = [
		async_qa_gen_llama_index(
			content, llm, prompt, question_num_per_content, max_retries
		)
		for content, prompt in zip(
			final_df["content"].tolist(), final_df["prompt"].tolist()
		)
	]

	loops = get_event_loop()
	results = loops.run_until_complete(process_batch(tasks, batch))

	return results


async def async_qa_gen_llama_index(
	content: str,
	llm: LLM,
	prompt: str,
	question_num: int = 1,
	max_retries: int = 3,
):
	"""
	Generate a qa set by using the given content and the llama index model.
	You must select the question type.

	:param content: Content string
	:param llm: Llama index model
	:param prompt: The prompt to use for the qa generation.
	    The prompt must include the following placeholders:
	    - {{text}}: The content string
	    - {{num_questions}}: The number of questions to generate
	:param question_num: The number of questions to generate
	:param max_retries: Maximum number of retries when generated question number is not equal to the target number
	:return: List of dictionaries containing the query and generation_gt
	"""
	validate_llama_index_prompt(prompt)

	async def generate(content: str, llm: LLM):
		for _ in range(max_retries):
			output = await llm.acomplete(
				prompt.replace("{{text}}", content).replace(
					"{{num_questions}}", str(question_num)
				)
			)
			result = parse_output(output.text)
			if len(result) == question_num:
				return result
		raise InterruptedError(
			f"Failed to generate output of length {question_num} after {max_retries} retries."
		)

	return await generate(content, llm)


async def generate_basic_answer(llm: LLM, passage_str: str, query: str) -> str:
	basic_answer_system_prompt = """You are an AI assistant to answer the given question in the provide evidence text.
    You can find the evidence from the given text about question, and you have to write a proper answer to the given question.
    You have to preserve the question's language at the answer.
    For example, if the input question is Korean, the output answer must be in Korean.
    """
	user_prompt = f"Text:\n<|text_start|>\n{passage_str}\n<|text_end|>\n\nQuestion:\n{query}\n\nAnswer:"

	response = await llm.achat(
		messages=[
			ChatMessage(role=MessageRole.SYSTEM, content=basic_answer_system_prompt),
			ChatMessage(role=MessageRole.USER, content=user_prompt),
		],
		temperature=1.0,
	)
	return response.message.content


def validate_llama_index_prompt(prompt: str) -> bool:
	"""
	Validate the prompt for the llama index model.
	The prompt must include the following placeholders:
	- {{text}}: The content string
	- {{num_questions}}: The number of questions to generate
	"""
	if "{{text}}" not in prompt:
		raise ValueError("The prompt must include the placeholder {{text}}.")
	if "{{num_questions}}" not in prompt:
		raise ValueError("The prompt must include the placeholder {{num_questions}}.")
	return True


def parse_output(result: str) -> List[Dict]:
	result = result.strip()
	result = result.split("[Q]:")
	final_result = list()
	for res in result:
		res = res.strip()
		if res and "\n[A]:" in res:
			qa = res.split("\n[A]:")
			final_result.append(
				{"query": qa[0].strip(), "generation_gt": qa[1].strip()}
			)
	return final_result


def distribute_list_by_ratio(input_list, ratio) -> List[List[Any]]:
	total_ratio = sum(ratio)
	total_length = len(input_list)

	# Calculate the length of each slice
	slice_lengths = [int((r / total_ratio) * total_length) for r in ratio]

	# Adjust the last slice in case of rounding issues
	slice_lengths[-1] = total_length - sum(slice_lengths[:-1])

	slices = []
	start = 0
	for length in slice_lengths:
		end = start + length
		slices.append(input_list[start:end])
		start = end

	return slices
