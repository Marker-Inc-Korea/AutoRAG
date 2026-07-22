from llama_index.core.base.llms.types import ChatMessage, MessageRole

FILTER_PROMPT = {
	"dontknow_filter": {
		"en": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""The following sentence is an answer about a question. You have to decide the answer implies 'I don't know'.
If the answer implies 'I don't know', return True. If not, return False.""",
			),
		],
		"ko": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""다음 문장은 어떠한 질문에 대한 대답입니다. 해당 문장이 질문에 대해서 '모른다고' 답한 것인지 판단하십시오.
만약 해당 문장이 '모른다고' 답한 것이라면, True를 반환하세요. 그렇지 않다면 False를 반환하세요.""",
			)
		],
		"ja": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""次の文章はある質問に対する答えです。 該当文章が質問に対して「知らない」と答えたのか判断します。
もし、その文章が「知らない」と答えたのであれば、Trueを返します。 そうでなければFalseを返します。""",
			)
		],
	},
	"passage_dependency": {
		"en": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""You are a classifier that recognize 'passage dependent' questions.
The 'passage dependent' is the question that the answer will be change depending on what passage you choose.
For example) 'What is the highest score according to the table?'
This sentence is the passage dependent question because the answer will be different depending on the table.

In contrast, the following sentence is not passage dependant.
'What is the highest score of the KBO baseball history in one game?'
'What is the capital of France?'
These sentences will have the same answer regardless of the passage.

Please return True if the input question is passage dependent. Else return False.""",
			)
		],
		"ko": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""당신은 '단락 의존' 질문을 인식하는 분류기입니다.
'단락 의존'이란 어떤 단락이 선택 되는지 따라 답이 달라지는 질문을 의미합니다.
예를 들어, '주어진 표에 따르면 가장 높은 점수는 무엇인가요?'라는 질문은 단락 의존 질문입니다. 왜냐하면 표가 어떤 것인지에 따라 그 답이 달라지기 때문입니다.

반면에, 다음 문장들은 단락 의존적이지 않습니다.
'KBO 야구 역사상 한 경기에서 가장 높은 점수는 무엇인가요?' 또는 '프랑스의 수도는 무엇인가요?'
이러한 문장은 단락에 관계 없이 동일한 답을 가집니다.

입력된 질문이 단락 의존적이라면 True를 반환하고, 그렇지 않으면 False를 반환하세요.""",
			)
		],
		"ja": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""あなたは「段落依存」の質問を認識する分類器です。
「段落依存」とは、どの段落が選択されるかによって答えが変わる質問を意味します。
たとえば、「与えられた表によると、最も高い点数は何ですか？」という質問は、段落依存の質問です。 なぜなら、表がどんなものかによってその答えが変わるからです。

一方、次の文章は段落依存的ではありません。
KBO野球史上1試合で最も高い点数は何ですか?またはフランスの首都は何ですか?'
このような文章は段落に関係なく同じ答えを持ちます。

入力された質問が段落依存的である場合はTrueを返し、そうでない場合はFalseを返します。""",
			)
		],
	},
}
