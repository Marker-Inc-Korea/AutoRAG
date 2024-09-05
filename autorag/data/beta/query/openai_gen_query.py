import itertools
from typing import Dict

from openai import AsyncClient
from pydantic import BaseModel

from autorag.data.beta.query.prompt import factoid_single_hop_system_prompt


class Response(BaseModel):
	query: str


# Single hop QA generation OpenAI
async def query_gen_openai_base(
	row: Dict,
	client: AsyncClient,
	system_prompt: str,
	model_name: str = "gpt-4o-2024-08-06",
):
	context = list(itertools.chain.from_iterable(row["retrieval_gt_contents"]))
	context_str = "Text:\n" + "\n".join(
		[f"{i + 1}. {c}" for i, c in enumerate(context)]
	)
	user_prompt = f"{context_str}\n\nGenerated Question from the Text:\n"

	completion = await client.beta.chat.completions.parse(
		model=model_name,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		response_format=Response,
	)
	row["query"] = completion.choices[0].message.parsed.query
	return row


async def factoid_query_gen(
	row: Dict, client: AsyncClient, model_name: str = "gpt-4o-2024-08-06"
) -> Dict:
	return await query_gen_openai_base(
		row, client, factoid_single_hop_system_prompt, model_name
	)
