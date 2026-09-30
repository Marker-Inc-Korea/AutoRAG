import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactElement } from "react";
import type { AnswerPhase, ChatAttachment, ChatSummary, CitationEvidence, SearchStreamEvent } from "../../../shared/search-contract";
import type { SearchBridge } from "../../../shared/search-contract";
import { parseAnswerBlocks, parseInline, type AnswerBlock } from "../state/answer-markdown";
import { groupChatHistory, historyEmptyText } from "../state/history";
import { CloseIcon, HistoryIcon, PlusIcon, SearchIcon, SendIcon } from "./icons";

const autorag = (window as unknown as { readonly autorag: { readonly search: SearchBridge } }).autorag;

export function AiSearchPanel({
	selectedEvidence,
	onSelectEvidence,
	onPublish,
}: {
	readonly selectedEvidence: number | null;
	readonly onSelectEvidence: (selected: number) => void;
	readonly onPublish: (evidence: readonly CitationEvidence[], sessionId: string | null) => void;
}): ReactElement {
	const [query, setQuery] = useState("");
	const [quick, setQuick] = useState<AnswerPhase | null>(null);
	const [deep, setDeep] = useState<AnswerPhase | null>(null);
	const [quickDraft, setQuickDraft] = useState("");
	const [deepDraft, setDeepDraft] = useState("");
	const [progress, setProgress] = useState("");
	const [searchId, setSearchId] = useState<string | null>(null);
	const [sessionId, setSessionId] = useState<string | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [stopped, setStopped] = useState(false);
	const [submittedQuery, setSubmittedQuery] = useState<string | null>(null);
	const [chatId, setChatId] = useState<string | null>(null);
	const [historyOpen, setHistoryOpen] = useState(false);
	const [historyQuery, setHistoryQuery] = useState("");
	const [history, setHistory] = useState<readonly ChatSummary[]>([]);
	const [attachments, setAttachments] = useState<readonly ChatAttachment[]>([]);
	const queryRef = useRef(query);
	const historySeq = useRef(0);

	useEffect(() => {
		queryRef.current = query;
	}, [query]);

	useEffect(() => {
		const unsubscribe = autorag.search.onEvent?.((event: SearchStreamEvent) => {
			if (event.searchId !== searchId) return;
			switch (event.type) {
				case "progress":
					setProgress(event.text);
					break;
				case "quick-delta":
					setQuickDraft((current) => current + event.text);
					break;
				case "deep-delta":
					setDeepDraft((current) => current + event.text);
					break;
				case "quick":
					setSessionId(event.sessionId);
					setQuick(event.phase);
					setQuickDraft(event.phase.answer);
					setProgress("");
					break;
				case "deep":
					setSessionId(event.sessionId);
					setDeep(event.phase);
					setDeepDraft(event.phase.answer);
					setProgress("");
					break;
				case "error":
					setError(event.message);
					setProgress("");
					setSearchId(null);
					break;
				case "cancelled":
					setStopped(true);
					setProgress("");
					setSearchId(null);
					break;
			}
		});
		return unsubscribe;
	}, [searchId]);

	const evidence = useMemo(() => deep?.evidence ?? quick?.evidence ?? [], [deep, quick]);

	useEffect(() => {
		onPublish(evidence, sessionId);
	}, [evidence, sessionId, onPublish]);

	const historyGroups = useMemo(() => groupChatHistory(history, new Date()), [history]);

	useEffect(() => {
		const onKeyDown = (event: globalThis.KeyboardEvent): void => {
			const meta = event.metaKey || event.ctrlKey;
			if (meta && event.shiftKey && event.key.toLowerCase() === "h") {
				event.preventDefault();
				toggleHistory();
				return;
			}
			if (historyOpen) return;
			if (meta && !event.shiftKey && !event.altKey && event.key.toLowerCase() === "n") {
				event.preventDefault();
				newChat();
			}
		};
		window.addEventListener("keydown", onKeyDown);
		return () => window.removeEventListener("keydown", onKeyDown);
	});

	async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
		event.preventDefault();
		const text = query.trim();
		if (text.length === 0 || searchId !== null) return;
		const nextChatId = crypto.randomUUID();
		const nextSearchId = crypto.randomUUID();
		setSearchId(nextSearchId);
		setQuick(null);
		setDeep(null);
		setQuickDraft("");
		setDeepDraft("");
		setSessionId(null);
		setError(null);
		setStopped(false);
		setSubmittedQuery(text);
		setProgress("Reviewing the query.");
		setChatId(nextChatId);
		await autorag.search.start(nextSearchId, nextChatId, text, attachments);
		setAttachments([]);
	}

	async function openHistory(): Promise<void> {
		historySeq.current += 1;
		const seq = historySeq.current;
		const items = await autorag.search.historyList();
		if (seq !== historySeq.current) return;
		setHistory(items);
		setHistoryQuery("");
		setHistoryOpen(true);
	}

	function closeHistory(): void {
		setHistoryOpen(false);
	}

	function toggleHistory(): void {
		if (historyOpen) {
			closeHistory();
			return;
		}
		void openHistory();
	}

	/** The store owns the matching (title + first Quick answer); a stale reply never lands after a newer keystroke. */
	async function searchHistory(value: string): Promise<void> {
		setHistoryQuery(value);
		historySeq.current += 1;
		const seq = historySeq.current;
		const items = await autorag.search.historySearch(value);
		if (seq !== historySeq.current) return;
		setHistory(items);
	}

	function newChat(): void {
		setQuick(null);
		setDeep(null);
		setQuickDraft("");
		setDeepDraft("");
		setError(null);
		setStopped(false);
		setProgress("");
		setQuery("");
		setSubmittedQuery(null);
		setAttachments([]);
		setChatId(null);
		setHistoryOpen(false);
	}

	async function loadHistory(id: string): Promise<void> {
		const record = await autorag.search.historyGet(id);
		const assistant = record?.messages.findLast((message) => message.role === "assistant");
		if (assistant?.role !== "assistant") return;
		const user = record?.messages.findLast((message) => message.role === "user");
		setQuick(assistant.quick);
		setDeep(assistant.deep);
		setQuickDraft(assistant.quick?.answer ?? "");
		setDeepDraft(assistant.deep?.answer ?? "");
		setSessionId(assistant.sessionId);
		setSubmittedQuery(user?.role === "user" ? user.text : null);
		setStopped(assistant.stopped);
		setError(null);
		setProgress("");
		setChatId(id);
		setHistoryOpen(false);
	}

	async function stop(): Promise<void> {
		if (searchId === null) return;
		await autorag.search.cancel(searchId);
		setSearchId(null);
		setProgress("");
		setStopped(true);
	}

	return (
		<section className="ai" aria-label="AI Search">
			<div className="ai__header">
				<span className="ai__title">AI Search</span>
				<div className="ai__header-actions">
					<button type="button" className={`icon-button${historyOpen ? " icon-button--active" : ""}`} title="Chat history ⌘⇧H" aria-label="Chat history" onClick={() => void toggleHistory()}><HistoryIcon /></button>
					<button type="button" className="icon-button" title="New chat ⌘N" aria-label="New chat" onClick={newChat}><PlusIcon /></button>
				</div>
			</div>
			{historyOpen ? (
				<>
					<div className="ai__history-scrim" aria-hidden="true" onClick={closeHistory} />
					<div className="ai__history-popover" aria-label="Chat history">
						<div className="ai__history-header">
							<SearchIcon className="ai__history-search-icon" />
							<input
								autoFocus
								value={historyQuery}
								onChange={(event) => void searchHistory(event.target.value)}
								onKeyDown={(event) => {
									if (event.key === "Escape") {
										event.preventDefault();
										event.stopPropagation();
										closeHistory();
										return;
									}
									if (event.key !== "Enter") return;
									event.preventDefault();
									const first = history[0];
									if (first !== undefined) void loadHistory(first.id);
								}}
								placeholder="대화 기록 검색"
								aria-label="대화 기록 검색"
							/>
							<button type="button" className="icon-button" aria-label="Close history" onClick={closeHistory}><CloseIcon size={12} /></button>
						</div>
						<div className="ai__history-list">
							{historyGroups.length === 0 ? (
								<span className="ai__history-empty">{historyEmptyText(historyQuery)}</span>
							) : (
								historyGroups.map((group) => (
									<div key={group.label}>
										<div className="ai__history-group">{group.label}</div>
										{group.items.map((item) => (
											<button
												type="button"
												key={item.id}
												className={`ai__history-item${item.id === chatId ? " ai__history-item--current" : ""}`}
												onClick={() => void loadHistory(item.id)}
											>
												<span className="ai__history-dot" />
												<span className="ai__history-text">
													<strong>{item.title}</strong>
													<small>{item.snippet}</small>
												</span>
												<span className="ai__history-time">{item.time}</span>
											</button>
										))}
									</div>
								))
							)}
						</div>
					</div>
				</>
			) : null}
			<div className="ai__body">
				<div className="ai__messages">
					{quick === null && deep === null && error === null && submittedQuery === null ? (
						<div className="ai__empty">
							<div className="ai__empty-title">무엇을 찾아드릴까요?</div>
							<div className="ai__empty-sub">
								파일, 메일, Slack, Notion 전체에서 찾아 빠른 답변과 정확한 답변을 함께 드립니다.
							</div>
						</div>
					) : null}
					{submittedQuery === null ? null : <div className="ai__user-message"><span>You</span><p>{submittedQuery}</p></div>}
					{submittedQuery === null ? null : <div className="ai__assistant-label">Assistant</div>}
					{error === null ? null : <div className="ai__error">{error}</div>}
					{submittedQuery === null ? null : <AnswerCard label="Quick" phase={quick} draft={quickDraft} pending={searchId !== null && quick === null} selectedEvidence={selectedEvidence} onCitation={onSelectEvidence} />}
					{submittedQuery === null ? null : <AnswerCard label="Deep" phase={deep} draft={deepDraft} pending={searchId !== null && deep === null} selectedEvidence={selectedEvidence} onCitation={onSelectEvidence} />}
					{progress === "" ? null : <div className="ai__progress">{progress}</div>}
					{stopped ? <div className="ai__stopped">응답 생성을 중단했습니다.</div> : null}
				</div>
				<form
					className="ai__composer"
					onSubmit={(event) => void submit(event)}
					onDragOver={(event) => event.preventDefault()}
					onDrop={(event) => {
						event.preventDefault();
						const dropped = [...event.dataTransfer.files].map((file) => ({
							kind: "file" as const,
							id: (file as File & { readonly path?: string }).path ?? file.name,
							name: file.name,
						}));
						setAttachments((current) => [...current, ...dropped]);
					}}
				>
					{attachments.length > 0 ? <span className="ai__attachments">{attachments.map((item) => item.name).join(", ")}</span> : null}
					<div className="ai__composer-row">
						<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="파일, 메일, 메신저 전체에서 물어보세요" aria-label="Ask AutoRAG" disabled={searchId !== null} />
						{searchId === null ? <button type="submit" className="ai__send-button" title="Ask AutoRAG" aria-label="Ask AutoRAG" disabled={query.trim().length === 0}><SendIcon /></button> : <button type="button" className="ai__send-button ai__send-button--stop" title="Stop search" aria-label="Stop search" onClick={() => void stop()}><span /></button>}
					</div>
				</form>
			</div>
		</section>
	);
}

function AnswerCard({
	label,
	phase,
	draft,
	pending,
	selectedEvidence,
	onCitation,
}: {
	readonly label: string;
	readonly phase: AnswerPhase | null;
	readonly draft: string;
	readonly pending: boolean;
	readonly selectedEvidence: number | null;
	readonly onCitation: (selected: number) => void;
}): ReactElement {
	const answerText = phase !== null ? phase.answer : draft;
	if (answerText === "") {
		return (
			<article className={`ai__answer ai__answer--${label.toLowerCase()} ai__answer--pending`}>
				<div className="ai__answer-heading"><span className="ai__answer-label"><i />{label}</span><span>{pending ? "Searching…" : "Stopped"}</span></div>
				<div className="ai__skeleton"><span /><span /><span /></div>
			</article>
		);
	}
	const streaming = phase === null;
	return (
		<article className={`ai__answer ai__answer--${label.toLowerCase()}${streaming ? " ai__answer--streaming" : ""}`}>
			<div className="ai__answer-heading">
				<span className="ai__answer-label"><i />{label}</span>
				<span>{phase !== null ? phase.meta : "Streaming…"}</span>
			</div>
			<div className="ai__answer-content">
				{parseAnswerBlocks(answerText).map((block, index) => (
					<AnswerBlockView key={`${index}`} block={block} selected={selectedEvidence} onCitation={onCitation} />
				))}
			</div>
		</article>
	);
}

function AnswerBlockView({
	block,
	selected,
	onCitation,
}: {
	readonly block: AnswerBlock;
	readonly selected: number | null;
	readonly onCitation: (selected: number) => void;
}): ReactElement {
	const segments = parseInline(block.text).map((segment, index) => {
		switch (segment.kind) {
			case "bold":
				return <strong key={`${index}`}>{segment.text}</strong>;
			case "code":
				return <code key={`${index}`} className="ai__inline-code">{segment.text}</code>;
			case "citation":
				return (
					<button
						type="button"
						key={`${index}`}
						className={`ai__citation${segment.number === selected ? " ai__citation--active" : ""}`}
						onClick={() => onCitation(segment.number)}
					>
						{segment.number}
					</button>
				);
			default:
				return <span key={`${index}`}>{segment.text}</span>;
		}
	});
	switch (block.kind) {
		case "heading":
			return <><div className="ai__block ai__block--heading">{segments}</div></>;
		case "list":
			return (
				<div className="ai__block ai__block--list">
					<span className="ai__block-marker">–</span>
					<span>{segments}</span>
				</div>
			);
		case "quote":
			return <div className="ai__block ai__block--quote">{segments}</div>;
		case "code":
			return <pre className="ai__block ai__block--code">{block.text}</pre>;
		default:
			return <p className="ai__block ai__block--paragraph">{segments}</p>;
	}
}
