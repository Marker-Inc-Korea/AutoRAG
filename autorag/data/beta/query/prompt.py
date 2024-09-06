from llama_index.core.base.llms.types import ChatMessage, MessageRole

QUERY_GEN_PROMPT = {
	"factoid_single_hop": {
		"en": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""You're an AI tasked to convert Text into a factoid question.
Factoid questions are those seeking brief, factual information that can be easily verified. They typically require a yes or no answer or a brief explanation and often inquire about specific details such as dates, names, places, or events.

Examples of factoid questions include:

- What is the capital of France?
- Who invented the light bulb?
- When was Wikipedia founded?

Instructions:
1. Questions MUST BE extracted from given Text
2. Questions should be as detailed as possible from Text
3. Create questions that ask about factual information from the Text
4. Do not mention any of these in the questions: "in the given text", "in the provided information", etc.
Users do not know the passage source of the question, so it should not be mentioned in the question.""",
			)
		],
		"ko": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""당신은 주어진 Text를 '사실 질문'으로 변환하는 AI입니다.

사실 질문(factoid questions)이란 사실적인 정보를 요구하는 질문으로, 쉽게 검증할 수 있는 답변을 필요로 합니다. 일반적으로 예/아니오 답변이나 간단한 설명을 요구하며, 날짜, 이름, 장소 또는 사건과 같은 구체적인 세부사항에 대해 묻는 질문입니다.

사실 질문의 예는 다음과 같습니다:

	•	프랑스의 수도는 어디입니까?
	•	전구를 발명한 사람은 누구입니까?
	•	위키피디아는 언제 설립되었습니까?

지침:
	1.	질문은 반드시 주어진 Text를 기반으로 작성되어야 합니다.
	2.	질문은 Text를 기반으로 가능한 한 구체적으로 작성되어야 합니다.
	3.	Text에서 사실적 정보를 요구하는 질문을 만들어야 합니다. 즉, Text를 기반으로 사실 질문을 만드세요.
	4.	질문에 “주어진 Text에서” 또는 “제공된 단락에서”와 같은 표현을 포함해서는 안 됩니다.
사용자는 질문의 출처가 Text라는 것을 모르기 때문에 반드시 그 출처를 언급해서는 안 됩니다.
	5.	질문을 한국어로 작성하세요.""",
			)
		],
	},
	"concept_completion": {
		"en": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""You're an AI tasked to convert Text into a "Concept Completion" Question.
A “concept completion” question asks directly about the essence or identity of a concept.

Follow the following instructions.
Instructions:
1. Questions MUST BE extracted from given Text
2. Questions should be as detailed as possible from Text
3. Create questions that ask about information from the Text
4. MUST include specific keywords from the Text.
5. Do not mention any of these in the questions: "in the given text", "in the provided information", etc.
Users do not know the passage source of the question, so it should not be mentioned in the question.""",
			)
		],
		"ko": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""당신은 Text를 “개념 완성” 질문으로 변환하는 AI입니다.
"개념 완성" 질문은 개념의 본질이나 정체성에 대해 직접적으로 묻는 질문입니다.

다음 지시사항을 따르세요.
지시사항:
1.	질문은 반드시 주어진 Text를 기반으로 작성되어야 합니다.
2.	질문은 Text를 기반으로 가능한 한 자세하게 작성되어야 합니다.
3.	Text에서 제공된 정보를 묻는 질문을 생성하세요.
4.	Text의 특정 키워드를 반드시 질문에 포함하세요.
5.	질문에 “주어진 Text에서” 또는 “제공된 단락에서”와 같은 표현을 포함해서는 안 됩니다.
사용자는 질문의 출처가 Text라는 것을 모르기 때문에 반드시 그 출처를 언급해서는 안 됩니다.
6.	질문을 한국어로 작성하세요.""",
			)
		],
	},
	"two_hop_incremental": {
		"en": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="Generate a multi-hop question for the given answer which requires reference to all of the given documents.",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Document 1: The Municipality of Nuevo Laredo is located in the Mexican state of Tamaulipas.
Document 2: The Ciudad Deportiva (Sports City ¨ ¨) is a sports
complex in Nuevo Laredo, Mexico. It is home to the Tecolotes de
Nuevo Laredo Mexican Baseball League team and ...""",
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="""Answer: Tamaulipas
One-hop question (using Document 1): In which Mexican state is Nuevo Laredo located?
Two-hop question (using Document 2):  In which Mexican state can one find the Ciudad Deportiva, home to the Tecolotes de Nuevo Laredo?""",
			),
		],
		"ko": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="Generate a multi-hop question for the given answer which requires reference to all of the given documents.",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Document 1: The Municipality of Nuevo Laredo is located in the Mexican state of Tamaulipas.
Document 2: The Ciudad Deportiva (Sports City ¨ ¨) is a sports
complex in Nuevo Laredo, Mexico. It is home to the Tecolotes de
Nuevo Laredo Mexican Baseball League team and ...""",
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="""Answer: Tamaulipas
One-hop question (using Document 1): In which Mexican state is Nuevo Laredo located?
Two-hop question (using Document 2):  In which Mexican state can one find the Ciudad Deportiva, home to the Tecolotes de Nuevo Laredo?""",
			),
		],
	},
}
