/** Server-sent events of an OpenAI Responses API stream, as the hosted Dazzi `/v1/responses` emits them. */

const USAGE = {
	input_tokens: 10,
	output_tokens: 5,
	total_tokens: 15,
	input_tokens_details: { cached_tokens: 0 },
	output_tokens_details: { reasoning_tokens: 0 },
};

function sse(events: readonly Record<string, unknown>[]): string {
	return events.map((event) => `event: ${String(event.type)}\ndata: ${JSON.stringify(event)}\n\n`).join("");
}

function response(id: string, status: "in_progress" | "completed", output: unknown[]): Record<string, unknown> {
	return { id, object: "response", status, output, ...(status === "completed" ? { usage: USAGE } : {}) };
}

/** A streamed assistant text reply. */
export function responsesTextStream(text: string, id = "resp_text"): string {
	const item = {
		id: "msg_1",
		type: "message",
		status: "completed",
		role: "assistant",
		content: [{ type: "output_text", text }],
	};
	return sse([
		{ type: "response.created", response: response(id, "in_progress", []) },
		{
			type: "response.output_item.added",
			output_index: 0,
			item: { id: "msg_1", type: "message", status: "in_progress", role: "assistant", content: [] },
		},
		{
			type: "response.content_part.added",
			item_id: "msg_1",
			output_index: 0,
			content_index: 0,
			part: { type: "output_text", text: "" },
		},
		{ type: "response.output_text.delta", item_id: "msg_1", output_index: 0, content_index: 0, delta: text },
		{ type: "response.output_item.done", output_index: 0, item },
		{ type: "response.completed", response: response(id, "completed", [item]) },
	]);
}

/** A streamed assistant turn that calls one tool. */
export function responsesToolCallStream(name: string, args: Record<string, unknown>, id = "resp_tool"): string {
	const argumentsJson = JSON.stringify(args);
	const item = {
		id: "fc_1",
		type: "function_call",
		status: "completed",
		call_id: "call_1",
		name,
		arguments: argumentsJson,
	};
	return sse([
		{ type: "response.created", response: response(id, "in_progress", []) },
		{
			type: "response.output_item.added",
			output_index: 0,
			item: { id: "fc_1", type: "function_call", status: "in_progress", call_id: "call_1", name, arguments: "" },
		},
		{ type: "response.function_call_arguments.delta", item_id: "fc_1", output_index: 0, delta: argumentsJson },
		{ type: "response.function_call_arguments.done", item_id: "fc_1", output_index: 0, arguments: argumentsJson },
		{ type: "response.output_item.done", output_index: 0, item },
		{ type: "response.completed", response: response(id, "completed", [item]) },
	]);
}

export const SSE_HEADERS = { "content-type": "text/event-stream" } as const;
