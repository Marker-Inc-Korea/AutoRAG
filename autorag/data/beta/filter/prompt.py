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
	}
}
