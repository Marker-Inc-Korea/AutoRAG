import { useState, type ReactElement } from "react";
import type { CitationEvidence, SearchBridge } from "../../../shared/search-contract";
import { parseAnswerBlocks, parseInline, type AnswerBlock, type InlineSegment } from "../state/answer-markdown";
import {
	evidenceCrumbs,
	evidenceDetail,
	evidenceKind,
	feedbackToastText,
	nextEvidenceFeedback,
	type EvidenceFeedback,
} from "../state/evidence";
import { ChevronsUpIcon, EyeIcon, ThumbsDownIcon, ThumbsUpIcon } from "./icons";
import { FileTile } from "./primitives/FileTile";

const autorag = (window as unknown as { readonly autorag: { readonly search: SearchBridge } }).autorag;

export function EvidencePanel({
	evidence,
	sessionId,
	selected,
	open,
	onToggleOpen,
	onSelect,
	onQuickLook,
	onReveal,
	onToast,
}: {
	readonly evidence: readonly CitationEvidence[];
	readonly sessionId: string | null;
	readonly selected: number;
	readonly open: boolean;
	readonly onToggleOpen: () => void;
	readonly onSelect: (number: number) => void;
	readonly onQuickLook: (source: string) => void;
	readonly onReveal: (source: string) => void;
	readonly onToast: (text: string) => void;
}): ReactElement {
	const [votes, setVotes] = useState<Readonly<Record<string, EvidenceFeedback>>>({});
	const candidate = evidence.find((entry) => entry.number === selected) ?? evidence[0];
	if (candidate === undefined) {
		return <section className="evidence" aria-label="Evidence" hidden />;
	}
	const item = candidate;
	const voteKey = `${sessionId ?? "paused"}:${item.number}`;
	const vote = votes[voteKey] ?? null;

	function sendVote(pick: "up" | "down"): void {
		const next = nextEvidenceFeedback(vote, pick);
		setVotes((current) => ({ ...current, [voteKey]: next }));
		if (next === null || sessionId === null) return;
		void autorag.search
			.feedback(sessionId, next === "up" ? [item.number] : [], next === "down" ? [item.number] : [])
			.then(() => onToast(feedbackToastText(item.number, pick)))
			.catch((error: unknown) => onToast(error instanceof Error ? error.message : String(error)));
	}

	return (
		<section className={`evidence${open ? "" : " evidence--closed"}`} aria-label="Evidence">
			<div className="evidence__strip">
				<div className="evidence__tabs">
					{evidence.map((entry) => (
						<button
							type="button"
							key={entry.number}
							title={entry.title}
							className={`evidence__tab${entry.number === item.number ? " evidence__tab--active" : ""}`}
							onClick={() => onSelect(entry.number)}
						>
							{entry.number}
						</button>
					))}
				</div>
				<button
					type="button"
					className="evidence__toggle"
					title="Toggle evidence ⌘E"
					aria-label="Toggle evidence"
					onClick={onToggleOpen}
				>
					<ChevronsUpIcon />
				</button>
			</div>
			{open ? (
				<div className="evidence__view">
					<div className="evidence__header">
						<div className="evidence__title-row">
							<FileTile kind={evidenceKind(item.title)} size={22} />
							<span className="evidence__title">
								{item.number}. {item.title}
							</span>
							<div className="evidence__votes">
								<button
									type="button"
									title="이 근거가 도움이 됨"
									className={`evidence__vote${vote === "up" ? " evidence__vote--up-on" : ""}`}
									onClick={() => sendVote("up")}
								>
									<ThumbsUpIcon />
								</button>
								<button
									type="button"
									title="이 근거는 도움이 안 됨"
									className={`evidence__vote${vote === "down" ? " evidence__vote--down-on" : ""}`}
									onClick={() => sendVote("down")}
								>
									<ThumbsDownIcon />
								</button>
							</div>
							<button
								type="button"
								className="evidence__quicklook"
								disabled={item.source === null}
								onClick={() => {
									if (item.source !== null) onQuickLook(item.source);
								}}
							>
								<EyeIcon />
								Quick Look
							</button>
						</div>
						<div className="evidence__where">
							<button
								type="button"
								className="evidence__crumb"
								title="Show in Finder"
								disabled={item.source === null}
								onClick={() => {
									if (item.source !== null) onReveal(item.source);
								}}
							>
								{item.source === null
									? item.title
									: evidenceCrumbs(item.source).map((segment) => (
										<span key={segment.label} className="evidence__crumb-segment">
											{segment.last ? null : null}
											<span className={segment.last ? "evidence__crumb-last" : ""}>{segment.label}</span>
											{segment.last ? null : <span className="evidence__crumb-separator">›</span>}
										</span>
									))}
							</button>
							<span className="evidence__detail">{evidenceDetail(item.confidence)}</span>
						</div>
					</div>
					<div className="evidence__body">
						{item.summary.length === 0 ? null : (
							<div className="evidence__block evidence__block--paragraph">{renderInline(item.summary)}</div>
						)}
						{item.excerpts.map((excerpt, index) => (
							<EvidenceExcerpt key={`${index}`} text={excerpt} />
						))}
					</div>
				</div>
			) : null}
		</section>
	);
}

function EvidenceExcerpt({ text }: { readonly text: string }): ReactElement {
	return (
		<>
			{parseAnswerBlocks(text).map((block, index) => (
				<EvidenceBlock key={`${index}`} block={block} />
			))}
		</>
	);
}

function EvidenceBlock({ block }: { readonly block: AnswerBlock }): ReactElement {
	switch (block.kind) {
		case "heading":
			return (
				<div className="evidence__block evidence__block--heading">
					<span className="evidence__block-prefix">## </span>
					{renderInline(block.text)}
				</div>
			);
		case "list":
			return (
				<div className="evidence__block evidence__block--list">
					<span className="evidence__block-prefix">–</span>
					<span>{renderInline(block.text)}</span>
				</div>
			);
		case "quote":
			return <div className="evidence__block evidence__block--quote">{renderInline(block.text)}</div>;
		case "code":
			return <pre className="evidence__block evidence__block--code">{block.text}</pre>;
		default:
			return <div className="evidence__block evidence__block--paragraph">{renderInline(block.text)}</div>;
	}
}

function renderInline(text: string): readonly ReactElement[] {
	return parseInline(text).map((segment, index) => <InlineSegment key={`${index}`} segment={segment} />);
}

function InlineSegment({ segment }: { readonly segment: InlineSegment }): ReactElement {
	if (segment.kind === "bold") {
		return <strong>{segment.text}</strong>;
	}
	if (segment.kind === "code") {
		return <code className="evidence__inline-code">{segment.text}</code>;
	}
	return <>{segment.kind === "citation" ? `[${segment.number}]` : segment.text}</>;
}
