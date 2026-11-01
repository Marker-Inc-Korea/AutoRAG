import type { ExtensionFactory, ExtensionUIContext, Theme } from "@earendil-works/pi-coding-agent";
import { truncateToWidth } from "@earendil-works/pi-tui";

/** Where pi shows the progress widget: its own slot above the editor, where the submitted question lands. */
export const SEARCH_PROGRESS_WIDGET_KEY = "autorag-search-progress";

/** Frame interval of the spinner and of the elapsed-time redraw. */
const REFRESH_INTERVAL_MS = 100;
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] as const;
/** Header plus step lines; keeps the widget from pushing the chat out of a short terminal. */
const MAX_LINES = 8;

export type SearchProgressStatus = "running" | "done" | "failed";

/** One unit of work the search is doing (routing, one retrieval source, a model phase, …). */
export interface SearchProgressStep {
	readonly id: string;
	readonly label: string;
	readonly status: SearchProgressStatus;
	/** Outcome text: the result count, the route, or the failure's own message. */
	readonly detail?: string;
	readonly startedAt: number;
	readonly endedAt?: number;
}

/** An immutable view of the running search, handed to subscribers on every change. */
export interface SearchProgressSnapshot {
	readonly active: boolean;
	readonly query: string;
	readonly startedAt: number;
	/** A model turn streams: pi's own working loader is on screen then. */
	readonly modelStreaming: boolean;
	readonly steps: readonly SearchProgressStep[];
}

interface MutableStep {
	id: string;
	label: string;
	status: SearchProgressStatus;
	detail?: string;
	startedAt: number;
	endedAt?: number;
	/** Concurrent tracked jobs (one per search query) still in flight on this step. */
	pending: number;
	successes: number;
	failures: number;
	hits: number;
	counted: boolean;
	firstError?: string;
}

/** Options of {@link SearchProgressTracker.track}. */
export interface TrackOptions<T> {
	/** How many hits a settled job returned; the step reports the sum. */
	readonly count?: (value: T) => number;
}

/**
 * What the running search is doing right now, in the user's terms. The agent
 * reports into it; the TUI renders it so a silent first retrieval is visibly
 * alive. Reports outside a begun search are ignored, so the library path
 * (`searchDocuments()` with no UI) pays nothing.
 */
export class SearchProgressTracker {
	private active = false;
	private query = "";
	private startedAt = 0;
	private modelStreaming = false;
	/** Bumped by every {@link begin}: a job that settles after its search ended must not touch the next one. */
	private run = 0;
	private steps: MutableStep[] = [];
	private readonly listeners = new Set<(snapshot: SearchProgressSnapshot) => void>();

	private readonly now: () => number;

	constructor(now: () => number = Date.now) {
		this.now = now;
	}

	begin(query: string): void {
		this.run += 1;
		this.active = true;
		this.query = query;
		this.startedAt = this.now();
		this.modelStreaming = false;
		this.steps = [];
		this.emit();
	}

	end(): void {
		if (!this.active) return;
		this.active = false;
		this.modelStreaming = false;
		this.emit();
	}

	setModelStreaming(streaming: boolean): void {
		if (!this.active || this.modelStreaming === streaming) return;
		this.modelStreaming = streaming;
		this.emit();
	}

	/** Starts (or restarts) a step that the caller closes with {@link endStep}. */
	beginStep(id: string, label: string): void {
		if (!this.active) return;
		this.open(id, label);
		this.emit();
	}

	endStep(id: string, status: Exclude<SearchProgressStatus, "running">, detail?: string): void {
		if (!this.active) return;
		const step = this.steps.find((entry) => entry.id === id);
		if (step === undefined) return;
		step.status = status;
		step.endedAt = this.now();
		if (detail === undefined) delete step.detail;
		else step.detail = detail;
		this.emit();
	}

	/**
	 * Removes a step that turned out not to belong to this search: speculative work whose branch was not taken.
	 * A job still running under that id settles into nothing, so the step does not come back.
	 */
	discardStep(id: string): void {
		if (!this.active) return;
		const index = this.steps.findIndex((entry) => entry.id === id);
		if (index < 0) return;
		this.steps.splice(index, 1);
		this.emit();
	}

	/**
	 * Runs `work` as step `id`: running while it is in flight, done with
	 * `detail(value)` when it resolves, failed with the error's own message when
	 * it rejects (the rejection still propagates).
	 */
	async step<T>(
		id: string,
		label: string,
		work: () => Promise<T>,
		detail?: (value: T) => string | undefined,
	): Promise<T> {
		const run = this.run;
		this.beginStep(id, label);
		try {
			const value = await work();
			if (this.run === run) this.endStep(id, "done", detail?.(value));
			return value;
		} catch (error) {
			if (this.run === run) this.endStep(id, "failed", error instanceof Error ? error.message : String(error));
			throw error;
		}
	}

	/**
	 * Follows one job as part of step `id`. Jobs sharing an id (the same source
	 * searched for several queries) form one step that ends when the last one
	 * settles. The job's result and rejection pass through untouched.
	 */
	track<T>(id: string, label: string, work: Promise<T>, options: TrackOptions<T> = {}): Promise<T> {
		if (!this.active) return work;
		const run = this.run;
		const step = this.open(id, label, true);
		step.pending += 1;
		this.emit();
		const settle = (): void => {
			if (this.run !== run || !this.active) return;
			step.pending -= 1;
			if (step.pending > 0) {
				this.emit();
				return;
			}
			step.endedAt = this.now();
			if (step.failures > 0 && step.successes === 0) {
				step.status = "failed";
				if (step.firstError !== undefined) step.detail = step.firstError;
			} else {
				step.status = "done";
				const parts: string[] = [];
				if (step.counted) parts.push(`${step.hits} ${step.hits === 1 ? "hit" : "hits"}`);
				if (step.failures > 0) parts.push(`${step.failures} failed: ${step.firstError ?? "unknown error"}`);
				if (parts.length > 0) step.detail = parts.join(", ");
			}
			this.emit();
		};
		return work.then(
			(value) => {
				if (this.run === run) {
					step.successes += 1;
					if (options.count !== undefined) {
						step.counted = true;
						step.hits += options.count(value);
					}
				}
				settle();
				return value;
			},
			(error: unknown) => {
				if (this.run === run) {
					step.failures += 1;
					step.firstError ??= error instanceof Error ? error.message : String(error);
				}
				settle();
				throw error;
			},
		);
	}

	snapshot(): SearchProgressSnapshot {
		return {
			active: this.active,
			query: this.query,
			startedAt: this.startedAt,
			modelStreaming: this.modelStreaming,
			steps: this.steps.map(({ id, label, status, detail, startedAt, endedAt }) => ({
				id,
				label,
				status,
				startedAt,
				...(detail === undefined ? {} : { detail }),
				...(endedAt === undefined ? {} : { endedAt }),
			})),
		};
	}

	/** Notified after every change. Not called on subscribe: read {@link snapshot} for the current state. */
	subscribe(listener: (snapshot: SearchProgressSnapshot) => void): () => void {
		this.listeners.add(listener);
		return () => {
			this.listeners.delete(listener);
		};
	}

	private open(id: string, label: string, tracked = false): MutableStep {
		const existing = this.steps.find((entry) => entry.id === id);
		// A tracked step that is still running keeps collecting jobs; anything else starts over.
		if (existing !== undefined && tracked && existing.status === "running") return existing;
		const fresh: MutableStep = {
			id,
			label,
			status: "running",
			startedAt: this.now(),
			pending: 0,
			successes: 0,
			failures: 0,
			hits: 0,
			counted: false,
		};
		if (existing === undefined) this.steps.push(fresh);
		else this.steps[this.steps.indexOf(existing)] = fresh;
		return fresh;
	}

	private emit(): void {
		if (this.listeners.size === 0) return;
		const snapshot = this.snapshot();
		for (const listener of [...this.listeners]) listener(snapshot);
	}
}

/** `7s` under a minute, `1:01` after. */
export function formatElapsed(ms: number): string {
	const seconds = Math.max(0, Math.floor(ms / 1000));
	if (seconds < 60) return `${seconds}s`;
	return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

/** Colour hooks for {@link renderSearchProgress}; pi's theme supplies them, tests pass identity. */
export interface SearchProgressPalette {
	readonly accent: (text: string) => string;
	readonly muted: (text: string) => string;
	readonly success: (text: string) => string;
	readonly error: (text: string) => string;
}

/**
 * The widget's lines at time `now`. Pure so the TUI and the tests render the
 * same thing: a spinner header with the elapsed time, then one line per step.
 * More steps than fit keep every running step and the newest finished ones.
 */
export function renderSearchProgress(
	snapshot: SearchProgressSnapshot,
	now: number,
	palette: SearchProgressPalette,
	width: number,
	interruptKey: string,
): string[] {
	const frame = SPINNER_FRAMES[Math.floor(now / 80) % SPINNER_FRAMES.length] ?? SPINNER_FRAMES[0];
	const elapsed = formatElapsed(now - snapshot.startedAt);
	const header = snapshot.modelStreaming
		? palette.muted(`Searching… ${elapsed}`)
		: `${palette.accent(frame)} ${palette.muted(`Searching… ${elapsed}${interruptKey === "" ? "" : ` · ${interruptKey} to stop`}`)}`;

	const stepLine = (step: SearchProgressStep): string => {
		const duration = formatElapsed((step.endedAt ?? now) - step.startedAt);
		const icon =
			step.status === "running"
				? palette.accent(frame)
				: step.status === "done"
					? palette.success("✓")
					: palette.error("✗");
		const detail = step.detail === undefined ? "" : ` — ${step.detail}`;
		return `  ${icon} ${palette.muted(`${step.label}${detail} · ${duration}`)}`;
	};

	const stepBudget = MAX_LINES - 1;
	let shown = snapshot.steps;
	let hidden = 0;
	if (shown.length > stepBudget) {
		const room = stepBudget - 1;
		const running = snapshot.steps.filter((step) => step.status === "running").slice(-room);
		const chosen = new Set<SearchProgressStep>(running);
		for (let index = snapshot.steps.length - 1; index >= 0 && chosen.size < room; index--) {
			const step = snapshot.steps[index];
			if (step !== undefined) chosen.add(step);
		}
		shown = snapshot.steps.filter((step) => chosen.has(step));
		hidden = snapshot.steps.length - shown.length;
	}
	const lines = [header, ...shown.map(stepLine)];
	if (hidden > 0) lines.push(`  ${palette.muted(`… ${hidden} earlier ${hidden === 1 ? "step" : "steps"}`)}`);
	return lines.map((line) => truncateToWidth(line, width));
}

/**
 * Binds the tracker to pi's widget slot: the widget appears when a search
 * begins and disappears when it ends. `isCurrent` lets a replaced session's
 * late notifications be dropped.
 */
export function bindSearchProgressWidget(
	ui: Pick<ExtensionUIContext, "setWidget">,
	tracker: SearchProgressTracker,
	interruptKey: () => string,
): () => void {
	let shown = false;
	const sync = (snapshot: SearchProgressSnapshot): void => {
		if (snapshot.active === shown) return;
		shown = snapshot.active;
		if (!snapshot.active) {
			ui.setWidget(SEARCH_PROGRESS_WIDGET_KEY, undefined);
			return;
		}
		ui.setWidget(SEARCH_PROGRESS_WIDGET_KEY, (tui, theme: Theme) => {
			const palette: SearchProgressPalette = {
				accent: (text) => theme.fg("accent", text),
				muted: (text) => theme.fg("muted", text),
				success: (text) => theme.fg("success", text),
				error: (text) => theme.fg("error", text),
			};
			// pi redraws only on events; a search that is merely waiting emits none, so tick.
			const timer = setInterval(() => tui.requestRender(), REFRESH_INTERVAL_MS);
			timer.unref();
			const unsubscribe = tracker.subscribe(() => tui.requestRender());
			return {
				render: (width: number): string[] =>
					renderSearchProgress(tracker.snapshot(), Date.now(), palette, width, interruptKey()),
				invalidate: (): void => undefined,
				dispose: (): void => {
					clearInterval(timer);
					unsubscribe();
				},
			};
		});
	};
	const unsubscribe = tracker.subscribe(sync);
	sync(tracker.snapshot());
	return () => {
		unsubscribe();
		if (shown) {
			shown = false;
			ui.setWidget(SEARCH_PROGRESS_WIDGET_KEY, undefined);
		}
	};
}

/**
 * pi extension that shows the tracker as a widget while a search runs. The
 * widget slot is rebound on every `session_start` (resume, `/new`, fork) and
 * released on shutdown, so a replaced session is never written to.
 */
export function createSearchProgressExtension(
	tracker: SearchProgressTracker,
	interruptKey: () => string,
): ExtensionFactory {
	return (pi) => {
		let release: (() => void) | undefined;
		pi.on("session_start", (_event, ctx) => {
			release?.();
			release = ctx.hasUI ? bindSearchProgressWidget(ctx.ui, tracker, interruptKey) : undefined;
		});
		pi.on("session_shutdown", () => {
			release?.();
			release = undefined;
		});
		// pi's own working loader owns the screen while a model turn streams.
		pi.on("agent_start", () => tracker.setModelStreaming(true));
		pi.on("agent_end", () => tracker.setModelStreaming(false));
	};
}
