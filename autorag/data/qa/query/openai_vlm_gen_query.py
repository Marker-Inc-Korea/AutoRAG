import base64
import itertools
from typing import Dict, List

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from openai import AsyncClient
from pydantic import BaseModel

import mimetypes
import aiofiles

from autorag.data.qa.query.prompt import QUERY_GEN_PROMPT


class Response(BaseModel):
	query: str


async def encode_image_async(image_path: str) -> str:
	async with aiofiles.open(image_path, "rb") as image_file:
		data = await image_file.read()
		return base64.b64encode(data).decode("utf-8")


# Single hop Visual QA generation OpenAI
async def query_gen_vlm_openai_base(
	row: Dict,
	client: AsyncClient,
	messages: List[ChatMessage],
	model_name: str = "gpt-4o-2024-08-06",
):
	context = list(itertools.chain.from_iterable(row.get("retrieval_gt_contents", [])))
	context_str = "Text:\n" + "\n".join(
		[f"{i + 1}. {c}" for i, c in enumerate(context)]
	)
	user_prompt = f"{context_str}\n\nGenerated Question from the Text and Image:\n"

	openai_messages = []
	for msg in messages:
		if msg.role == MessageRole.SYSTEM:
			openai_messages.append({"role": "system", "content": msg.content})

	user_content = [{"type": "text", "text": user_prompt}]

	if row.get("image_path"):
		mime_type, _ = mimetypes.guess_type(row["image_path"])
		mime_type = mime_type or "image/jpeg"

		base64_image = await encode_image_async(row["image_path"])
		user_content.append(
			{
				"type": "image_url",
				"image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
			}
		)

	openai_messages.append({"role": "user", "content": user_content})

	completion = await client.beta.chat.completions.parse(
		model=model_name,
		messages=openai_messages,
		response_format=Response,
	)
	row["query"] = completion.choices[0].message.parsed.query
	return row


async def vlm_factoid_query_gen(
	row: Dict,
	client: AsyncClient,
	model_name: str = "gpt-4o-2024-08-06",
	lang: str = "en",
) -> Dict:
	"""
	Create a visual factoid question using a vision-language model.
	The input row has to include an image_path.

	:return: The visual factoid question using openai vision model
	"""
	return await query_gen_vlm_openai_base(
		row, client, QUERY_GEN_PROMPT["factoid_single_hop"][lang], model_name
	)
