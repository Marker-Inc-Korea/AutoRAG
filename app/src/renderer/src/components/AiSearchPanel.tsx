import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactElement } from "react";
import type { AnswerPhase, ChatAttachment, ChatSummary, SearchStreamEvent } from "../../../shared/search-contract";
import type { SearchBridge } from "../../../shared/search-contract";
import { CloseIcon, HistoryIcon, PlusIcon, SearchIcon, SendIcon } from "./icons";

const autorag = (window as unknown as { readonly autorag: { readonly search: SearchBridge } }).autorag;

export function AiSearchPanel(): ReactElement {
	const [query, setQuery] = useState("");
	const [quick, setQuick] = useState<AnswerPhase | null>(null);
	const [deep, setDeep] = useState<AnswerPhase | null>(null);
	const [quickDraft, setQuickDraft] = useState("");
	const [deepDraft, setDeepDraft] = useState("");
	const [progress, setProgress] = useState("");
	const [searchId, setSearchId] = useState<string | null>(null);
	const [sessionId, setSessionId] = useState<string | null>(null);
	const [selectedEvidence, setSelectedEvidence] = useState<number | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [stopped, setStopped] = useState(false);
	const [submittedQuery, setSubmittedQuery] = useState<string | null>(null);
	const [historyOpen, setHistoryOpen] = useState(false);
	const [history, setHistory] = useState<readonly ChatSummary[]>([]);
	const [attachments, setAttachments] = useState<readonly ChatAttachment[]>([]);
	const queryRef = useRef(query);

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

	async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
		event.preventDefault();
		const text = query.trim();
		if (text.length === 0 || searchId !== null) return;
		const nextSearchId = crypto.randomUUID();
		setSearchId(nextSearchId);
		setQuick(null);
		setDeep(null);
		setQuickDraft("");
		setDeepDraft("");
		setSessionId(null);
		setSelectedEvidence(null);
		setError(null);
		setStopped(false);
		setSubmittedQuery(text);
		setProgress("Reviewing the query.");
		await autorag.search.start(nextSearchId, crypto.randomUUID(), text, attachments);
		setAttachments([]);
	}

	async function toggleHistory(): Promise<void> {
		if (!historyOpen) setHistory(await autorag.search.historyList());
		setHistoryOpen((open) => !open);
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
		setHistoryOpen(false);
	}

	async function stop(): Promise<void> {
		if (searchId === null) return;
		await autorag.search.cancel(searchId);
		setSearchId(null);
		setProgress("");
		setStopped(true);
	}

	async function sendFeedback(number: number, useful: boolean): Promise<void> {
		if (sessionId === null) return;
		await autorag.search.feedback(sessionId, useful ? [number] : [], useful ? [] : [number]);
	}

	return (
		<section className="ai" aria-label="AI Search">
			<div className="ai__header">
				<span className="ai__title">AI Search</span>
				<div className="ai__header-actions">
					<button type="button" className={`icon-button${historyOpen ? " icon-button--active" : ""}`} title="Chat history ⌘⇧H" aria-label="Chat history" onClick={() => void toggleHistory()}><HistoryIcon /></button>
					<button type="button" className="icon-button" title="New chat ⌘N" aria-label="New chat" onClick={() => { setQuick(null); setDeep(null); setQuickDraft(""); setDeepDraft(""); setError(null); setStopped(false); setProgress(""); setQuery(""); setSubmittedQuery(null); setHistoryOpen(false); }}><PlusIcon /></button>
				</div>
			</div>
			<div className="ai__body">
				{historyOpen ? (
					<div className="ai__history-popover" aria-label="Chat history">
						<div className="ai__history-header"><SearchIcon /><input autoFocus placeholder="대화 기록 검색" aria-label="대화 기록 검색" /><button type="button" className="icon-button" aria-label="Close history" onClick={() => setHistoryOpen(false)}><CloseIcon /></button></div>
						<div className="ai__history-list">
							{history.length === 0 ? <span className="ai__history-empty">저장된 대화가 없습니다.</span> : history.map((item) => (
								<button type="button" key={item.id} onClick={() => void loadHistory(item.id)}>
									<strong>{item.title}</strong><small>{item.snippet}</small>
								</button>
							))}
						</div>
					</div>
				) : null}
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
					{submittedQuery === null ? null : <AnswerCard label="Quick" phase={quick} draft={quickDraft} pending={searchId !== null && quick === null} onCitation={setSelectedEvidence} />}
					{submittedQuery === null ? null : <AnswerCard label="Deep" phase={deep} draft={deepDraft} pending={searchId !== null && deep === null} onCitation={setSelectedEvidence} />}
					{progress === "" ? null : <div className="ai__progress">{progress}</div>}
					{stopped ? <div className="ai__stopped">응답 생성을 중단했습니다.</div> : null}
					{selectedEvidence === null ? null : (
						<aside className="ai__evidence" aria-label="Evidence">
							<div className="ai__evidence-heading">Evidence [{selectedEvidence}]</div>
							{evidence
								.filter((item) => item.number === selectedEvidence)
								.map((item) => (
									<div className="ai__evidence-card" key={item.feedbackId}>
										<strong>{item.title}</strong>
										<p>{item.summary}</p>
										{item.excerpts.map((excerpt) => <blockquote key={excerpt}>{excerpt}</blockquote>)}
										<div className="ai__feedback">
											<button type="button" onClick={() => void sendFeedback(item.number, true)}>Useful</button>
											<button type="button" onClick={() => void sendFeedback(item.number, false)}>Not useful</button>
										</div>
									</div>
								))}
						</aside>
					)}
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
	onCitation,
}: {
	readonly label: string;
	readonly phase: AnswerPhase | null;
	readonly draft: string;
	readonly pending: boolean;
	readonly onCitation: (number: number) => void;
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
	const answer = answerText.replace(/\[(\d+)\]/gu, (_match, number: string) => ` [${number}] `);
	return (
		<article className={`ai__answer ai__answer--${label.toLowerCase()}${streaming ? " ai__answer--streaming" : ""}`}>
			<div className="ai__answer-heading">
				<span className="ai__answer-label"><i />{label}</span>
				<span>{phase !== null ? phase.meta : "Streaming…"}</span>
			</div>
			<div className="ai__answer-content">
				<p>
					{answer.split(/(\[\d+\])/gu).map((part, index) =>
						/^\[\d+\]$/u.test(part) ? (
							<button type="button" className="ai__citation" key={`${part}-${index}`} onClick={() => onCitation(Number(part.slice(1, -1)))}>
								{part}
							</button>
						) : (
							part
						),
					)}
				</p>
			</div>
		</article>
	);
}
