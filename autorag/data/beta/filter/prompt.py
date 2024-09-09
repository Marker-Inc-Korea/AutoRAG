from llama_index.core.base.llms.types import ChatMessage, MessageRole

FILTER_PROMPT = {
	"dontknow_filter": {
		"en": [ChatMessage(role=MessageRole.SYSTEM, content="")],
		"ko": [],
	}
}
