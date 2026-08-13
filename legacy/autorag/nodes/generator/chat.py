"""Helpers for truncating chat prompts without changing their message roles."""

from typing import Callable, Dict, List, Sequence, TypeVar, Union


Token = TypeVar("Token")
Encode = Callable[[str], Sequence[Token]]
Decode = Callable[[Sequence[Token]], str]
Prompt = Union[str, List[Dict]]


def messages_to_string(messages: List[Dict[str, str]]) -> str:
	"""Render messages only for token counting, never for model input."""
	formatted_parts = [
		f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>"
		for message in messages
	]
	formatted_parts.append("<|im_start|>assistant")
	return "\n".join(formatted_parts)


def count_message_tokens(messages: List[Dict], encode: Encode) -> int:
	"""Count the tokenized representation of a chat prompt."""
	return len(encode(messages_to_string(messages)))


def truncate_messages_by_token(
	messages: List[Dict], encode: Encode, decode: Decode, max_token_size: int
) -> List[Dict]:
	"""Fit a chat prompt in a token budget while preserving message structure."""
	truncated_messages = [dict(message) for message in messages]
	if count_message_tokens(truncated_messages, encode) <= max_token_size:
		return truncated_messages

	result: List[Dict] = []
	for message in truncated_messages:
		candidate = result + [message]
		if count_message_tokens(candidate, encode) <= max_token_size:
			result.append(message)
			continue

		content_tokens = encode(message.get("content", ""))
		low, high = 0, len(content_tokens)
		best_content = ""
		while low <= high:
			mid = (low + high) // 2
			truncated_message = {
				**message,
				"content": decode(content_tokens[:mid]),
			}
			candidate = result + [truncated_message]
			if count_message_tokens(candidate, encode) <= max_token_size:
				best_content = truncated_message["content"]
				low = mid + 1
			else:
				high = mid - 1

		if best_content or not result:
			result.append({**message, "content": best_content})
		break

	return result


def truncate_prompt_by_token(
	prompt: Prompt, encode: Encode, decode: Decode, max_token_size: int
) -> Prompt:
	"""Truncate text or chat prompts while retaining a chat prompt's roles."""
	if isinstance(prompt, list):
		return truncate_messages_by_token(prompt, encode, decode, max_token_size)
	tokens = encode(prompt)
	return decode(tokens[:max_token_size])
