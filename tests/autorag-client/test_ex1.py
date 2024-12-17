import os
import uuid

import pytest
from jinja2 import Template
from openai import OpenAI

from autorag_client import AutoRAGClient, RAGPipeline

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


root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
resources_dir = os.path.join(root_dir, "resources")


@pytest.mark.asyncio
async def test_setup_rag():
    """Setup RAG pipeline"""
    async with AutoRAGClient(api_key=os.environ.get("AUTORAG_API_KEY")) as client:
        # Setup project and upload document
        random_project_name = str(uuid.uuid4())
        description = "I am Havertz"
        project = await client.create_project(
            random_project_name, description=description
        )
        assert project.name == random_project_name
        assert project.description == description
        await project.upload_file(
            os.path.join(resources_dir, "parse_data", "all_files", "baseball_1.pdf")
        )

        # Initialize embedding and RAG pipeline with auto-configuration
        await project.embedding(vector_storage="auto")
        rag = await project.create_rag_pipeline(embedding_model="auto")

        assert isinstance(rag, RAGPipeline)


@pytest.mark.asyncio
async def test_query_rag():
    """Query RAG pipeline"""
    # Get retrievals from RAG
    question = "Who is Havertz?"
    rag_pipeline = RAGPipeline(
        client=AutoRAGClient(api_key=os.environ.get("AUTORAG_API_KEY")),
        project_id="proj_1",
    )
    async with AutoRAGClient(api_key=os.environ.get("AUTORAG_API_KEY")) as client:
        retrievals = await client.get_retrievals(rag_pipeline, question)

        # Generate prompt and setup OpenAI client
        prompt = RAG_PROMPT.render(
            context=retrievals.to_prompt_string(), question=question
        )
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
