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
    1. 다시 작성된 질문은 100자를 넘지 않아야 합니다. 가능한 경우 약어를 사용하십시오.
    2. 다시 작성된 질문은 합리적이어야 하며 사람이 이해하고 응답할 수 있어야 합니다.
    3. 다시 작성된 질문은 현재 Context에서 완전히 답변할 수 있어야 합니다.
    4. '제공된 글', '단락에 따르면?', 'Context에 의하면' 등의 문구는 질문에 나타날 수 없습니다.
    5. 한국어로 질문을 작성하세요.
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
		"ja": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""提供された質問に条件に関する内容を追加して、複雑さを高めます。
質問のContextに影響を与えるシナリオや条件を含めて、質問をより複雑にすることが目標です。
質問を再作成するときは、次のルールに従います。
    1. 再作成された質問は100文字を超えてはいけません。 可能であれば略語を使ってください
    2. 再作成された質問は合理的でなければならず、人が理解して回答できるものでなければなりません。
    3. 再作成された質問は、現在のContextで完全に答えられる必要があります。
    4. 「提供された文」、「段落によると?」、「Contextによると」などのフレーズは質問に表示されません。
    5. 日本語で質問を書きましょう。
""",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: 植物の根の機能は何ですか？
Context: 植物の根は土壌から水や栄養分を吸収し、植物を地面に固定し、栄養分を蓄えます。
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="植物の根は土壌栄養分と安定性に対してどのような役割をしますか？",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: ワクチンは病気をどのように予防しますか?
Context: ワクチンは、体の免疫反応を刺激して病原体を認識し、戦う抗体を生成することで病気から守ります。
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="ワクチンは体の免疫システムをどのように活用して病気を予防しますか？",
			),
		],
	},
	"reasoning_evolve_ragas": {
		"en": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""Complicate the given question by rewriting question into a multi-hop reasoning question based on the provided context.
Answering the question should require the reader to make multiple logical connections or inferences using the information available in given context.
Rules to follow when rewriting question:
1. Ensure that the rewritten question can be answered entirely from the information present in the contexts.
2. Do not frame questions that contains more than 15 words. Use abbreviation wherever possible.
3. Make sure the question is clear and unambiguous.
4. phrases like 'based on the provided context','according to the context',etc are not allowed to appear in the question.""",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: What is the capital of France?,
Context: France is a country in Western Europe. It has several cities, including Paris, Lyon, and Marseille. Paris is not only known for its cultural landmarks like the Eiffel Tower and the Louvre Museum but also as the administrative center.
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="Linking the Eiffel Tower and administrative center, which city stands as both?",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: What does the append() method do in Python?
Context: In Python, lists are used to store multiple items in a single variable. Lists are one of 4 built-in data types used to store collections of data. The append() method adds a single item to the end of a list.
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="If a list represents a variable collection, what method extends it by one item?",
			),
		],
		"ko": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""주어진 Context를 기반으로 기존 질문을 복잡하게 만들어 여러 논리적인 사고가 필요한 질문으로 다시 작성하세요.
질문에 답하려면 주어진 Context의 정보를 사용해 여러 논리적 사고나 추론을 해야 합니다.
질문을 다시 작성할 때 따라야 할 규칙:
1. 다시 작성된 질문은 Context에 있는 정보만으로 완전히 답변할 수 있어야 합니다.
2. 100자를 초과하는 질문을 작성하지 마세요. 가능한 경우 약어를 사용하세요.
3. 질문이 명확하고 모호하지 않도록 하세요.
4. '제공된 Context에 기반하여', '해당 단락에 따르면' 등의 문구는 질문에 포함되지 않아야 합니다.
5. 한국어로 질문을 작성하세요.""",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: 프랑스의 수도는 어디인가요?,
Context: 프랑스는 서유럽에 있는 나라입니다. 파리, 리옹, 마르세유를 포함한 여러 도시가 있습니다. 파리는 에펠탑과 루브르 박물관 같은 문화적 랜드마크로 유명할 뿐만 아니라 행정 중심지로도 알려져 있습니다.
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="에펠탑과 행정 중심지, 두 단어는 어떤 도시를 가리키나요?",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""질문: Python에서 append() 메서드는 무엇을 하나요?
컨텍스트: Python에서 리스트는 하나의 변수에 여러 항목을 저장하는 데 사용됩니다. 리스트는 데이터를 저장하는 데 사용되는 4가지 내장 데이터 유형 중 하나입니다. append() 메서드는 리스트의 끝에 새로운 항목을 추가합니다.
출력: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="리스트가 변수들을 모아 놓은 것을 나타낸다면, 어떤 메서드를 사용해야 항목을 하나 더 추가할 수 있습니까?",
			),
		],
		"ja": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""与えられたContextに基づいて既存の質問を複雑にして、様々な論理的思考が必要な質問として書き直しましょう。
質問に答えるためには、与えられたContextの情報を使って様々な論理的思考や推論をしなければなりません。
質問を再作成するときに従うべきルール:
1. 再作成された質問は、Contextにある情報だけで完全に答えられる必要があります。
2. 100文字を超える質問を作成してはいけません。 可能であれば略語を使ってください。
3. 質問が明確で曖昧にならないようにしましょう。
4. 「提供されたContextに基づいて」、「当該段落によると」などのフレーズは、質問に含まれてはいけません。
5. 日本語で質問を書きましょう。""",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: フランスの首都はどこですか？,
Context: フランスは西ヨーロッパにある国です。 パリ、リヨン、マルセイユを含むいくつかの都市があります。 パリはエッフェル塔やルーブル博物館のような文化的ランドマークとして有名なだけでなく、行政の中心地としても知られています。
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="エッフェル塔と行政の中心地、二つの単語はどんな都市を指していますか？",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: Pythonでappend() メソッドは何をしますか？
Context: Pythonで、リストは 1 つの変数に複数の項目を保存するために使用されます。 リストは、データを保存するために使用される 4 つの組み込みデータ タイプの 1 つです。 append()メソッドは、リストの最後に新しい項目を追加します。
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="リストが変数を集めたものである場合、どのメソッドを使えば項目を一つ追加することができますか？",
			),
		],
	},
	"compress_ragas": {
		"en": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""Rewrite the following question to make it more indirect and shorter while retaining the essence of the original question.
    The goal is to create a question that conveys the same meaning but in a less direct manner. The rewritten question should shorter so use abbreviation wherever possible.""",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: What is the distance between the Earth and the Moon?
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="How far is the Moon from Earth?",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: What ingredients are required to bake a chocolate cake?
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="What's needed for a chocolate cake?",
			),
		],
		"ko": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""주어진 질문을 더 간접적이고 짧게 다시 작성하세요.
        목표는 질문을 원래 질문의 본질을 유지하면서 너무 직설적이지 않게 만드는 것입니다.
        약어 등을 사용하여 질문을 더 짧게 만드세요.""",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: 지구와 달 사이의 거리는 얼마입니까?
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="달은 지구에서 얼마나 떨어져 있나요?",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: 초콜릿 케이크를 굽기 위해 필요한 재료는 무엇입니까?
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="초콜릿 케이크에 필요한 것은 무엇인가요?",
			),
		],
		"ja": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""与えられた質問をより間接的かつ短く書き換えます。
目標は、質問を元の質問の本質を保ちながら、あまりストレートにならないようにすることです。
略語などを使用して、質問をより短くします。""",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: 地球と月の間の距離はどれくらいですか？
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="月は地球からどれくらい離れていますか？",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Question: チョコレートケーキを焼くために必要な材料は何ですか？
Output: """,
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="チョコレートケーキに必要なものは何ですか？",
			),
		],
	},
}
