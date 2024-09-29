# The RAGAS prompts are coming from RAGAS under Apache-2.0 License. (English version) (the AutoRAG team translates Korean version prompt)
# You can see the original prompts at the RAGAS library at https://github.com/explodinggradients/ragas/blob/main/src/ragas/testset/prompts.py
from llama_index.core.base.llms.types import ChatMessage, MessageRole

QUERY_EVOLVE_PROMPT = {
	"conditional_evolve_ragas": {
		"en": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""Rewrite the provided question to increase its complexity by introducing a conditional element.
The goal is to make the question more intricate by incorporating a scenario or condition that affects the context of the question.
Follow the rules given below while rewriting the question.
    1. The rewritten question should not be longer than 25 words. Use abbreviation wherever possible.
    2. The rewritten question must be reasonable and must be understood and responded by humans.
    3. The rewritten question must be fully answerable from information present context.
    4. phrases like 'provided context','according to the context?',etc are not allowed to appear in the question.
""",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question : What is the function of the roots of a plant?
Context : The roots of a plant absorb water and nutrients from the soil, anchor the plant in the ground, and store food.
Output : """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="What dual purpose do plant roots serve concerning soil nutrients and stability?",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question : How do vaccines protect against diseases?
Context : Vaccines protect against diseases by stimulating the body's immune response to produce antibodies, which recognize and combat pathogens.
Output : """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="How do vaccines utilize the body's immune system to defend against pathogens?",
			),
		],
		"ko": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""제공된 질문에 조건에 관련한 내용을 추가하여 복잡성을 높이세요.
질문의 Context에 영향을 미치는 시나리오나 조건을 포함하여 질문을 더 복잡하게 만드는 것이 목표입니다.
질문을 다시 작성할 때 다음 규칙을 따르십시오.
    1. 다시 작성된 질문은 25자를 넘지 않아야 합니다. 가능한 경우 약어를 사용하십시오.
    2. 다시 작성된 질문은 합리적이어야 하며 사람이 이해하고 응답할 수 있어야 합니다.
    3. 다시 작성된 질문은 현재 Context에서 완전히 답변할 수 있어야 합니다.
    4. '제공된 글', '단락에 따르면?', 'Context에 의하면' 등의 문구는 질문에 나타날 수 없습니다.
""",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: 식물의 뿌리 기능이 뭐야?
Context: 식물의 뿌리는 토양에서 물과 영양분을 흡수하고, 식물을 땅에 고정하며, 영양분을 저장합니다.
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="식물의 뿌리는 토양 영양분과 안정성에 대해 어떤 역할을 하나요?",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: 백신은 질병을 어떻게 예방하나요?
Context: 백신은 신체의 면역 반응을 자극하여 병원체를 인식하고 싸우는 항체를 생성함으로써 질병으로부터 보호합니다.
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="백신은 신체의 면역 체계를 어떻게 활용해서 질병을 예방합니까?",
			),
		],
	},
}
