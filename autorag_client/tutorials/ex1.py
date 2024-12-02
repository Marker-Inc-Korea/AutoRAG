import asyncio
import os
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from autorag_client import AutoRAGClient

# Load environment variables from .env file
load_dotenv()


# Define RAG prompt template
RAG_PROMPT = Template("""
Answer the question based on the context below.

Context:
{{ context }}

Question: {{ question }}

Instructions:
- Answer the question using only the information from the context
- If you cannot find the answer in the context, say "I cannot answer based on the given context"
- Keep your answer concise and to the point

Answer?
""")


async def setup_rag():
	"""Setup RAG pipeline"""
	async with AutoRAGClient(api_key=os.environ.get("AUTORAG_API_KEY")) as client:
		# 1. Setup project and upload document
		project = await client.create_project("namename", description="jax")
		await project.upload_file("files/*.pdf")

		# Initialize embedding and RAG pipeline with auto-configuration
		await project.embedding(vector_storage="auto")
		rag = await project.create_rag_pipeline(embedding_model="auto")

		return rag


async def get_retrievals(rag_pipeline, question: str):
	"""Get retrievals from RAG"""
	async with AutoRAGClient(api_key=os.environ.get("AUTORAG_API_KEY")) as client:
		retrievals = await client.get_retrievals(rag_pipeline, question)
	return retrievals


async def generate_answer(retrievals, question: str):
	"""Generate answer using OpenAI"""
	# Generate prompt and setup OpenAI client
	prompt = RAG_PROMPT.render(context=retrievals.to_prompt_string(), question=question)
	client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

	# Request completion from GPT-4
	chat_completion = client.chat.completions.create(
		messages=[{"role": "user", "content": prompt}], model="gpt-4o-mini"
	)
	print("# Prompt: --------------------------\n", prompt)
	# 응답을 단일 객체로 감싸서 반환
	return type(
		"QueryResponse",
		(),
		{
			"question": question,
			"answer": chat_completion.choices[
				0
			].message.content,  # 실제 구현에서는 API 응답값 사용
			"retrievals": retrievals,
		},
	)()


async def main(question: str):
	# 1. Setup RAG pipeline
	rag = await setup_rag()

	# 2. Get retrievals from RAG pipeline
	retrievals = await get_retrievals(rag, question)
	print(f"Retrieved context: \n{retrievals.to_prompt_string()}")

	# 3. Generate answer
	response = await generate_answer(retrievals, question)
	print("# Response: --------------------------")
	print(f"Question: {question}")
	print(f"Answer: {response.answer}")


if __name__ == "__main__":
	asyncio.run(main(question="What is AutoRAG?"))
