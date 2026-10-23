import { existsSync, mkdirSync, unwatchFile, watchFile } from "node:fs";
import { join, resolve } from "node:path";
import parcelWatcher from "@parcel/watcher";
import { resolveAutoRAGHome } from "../config/home.ts";
import { acquireFileLock, type FileLockHandle } from "../filesystem/file-lock.ts";
import { type Automation, type AutomationEvent, loadAutomations, matchesAutomationPath } from "./config.ts";
import { type AutomationRunRecord, runAutomation } from "./run.ts";

export const DEFAULT_MAX_RUNS_PER_MINUTE = 20;
const RUNAWAY_WINDOW_MS = 60_000;
const CONFIG_POLL_INTERVAL_MS = 1000;
const PARCEL_IGNORED_SEGMENTS = ["**/.git/**", "**/node_modules/**", "**/.autorag/**"];

export interface AutomationSubscription {
	unsubscribe(): Promise<void>;
}

export type AutomationWatcherBackend = (
	dir: string,
	onEvents: (events: readonly AutomationEvent[]) => void,
	onError?: (error: Error) => void,
) => Promise<AutomationSubscription>;

export type AutomationRunner = (
	automation: Automation,
	events: readonly AutomationEvent[],
) => Promise<AutomationRunRecord>;

export interface AutomationDaemonOptions {
	readonly home?: string;
	readonly env?: NodeJS.ProcessEnv;
	readonly selfCommand?: readonly string[];
	readonly watcher?: AutomationWatcherBackend;
	readonly runner?: AutomationRunner;
	readonly maxRunsPerMinute?: number;
	readonly watchConfigFile?: boolean;
	readonly onRun?: (automation: Automation, record: AutomationRunRecord) => void;
	readonly onError?: (message: string) => void;
}

export interface AutomationDaemonHandle {
	reload(): Promise<void>;
	stop(): Promise<void>;
}

export class AutomationDaemonAlreadyRunningError extends Error {
	constructor(home: string) {
		super(`Automation daemon is already running for ${home}`);
		this.name = "AutomationDaemonAlreadyRunningError";
	}
}

export function automationDaemonLockPath(home: string = resolveAutoRAGHome()): string {
	return join(home, "automations", "daemon.lock");
}

export function isAutomationDaemonRunning(home: string = resolveAutoRAGHome()): boolean {
	const lockPath = automationDaemonLockPath(home);
	if (!existsSync(lockPath)) return false;
	try {
		const handle = acquireFileLock(lockPath, {
			timeoutMs: 0,
			staleMs: 0,
			timeoutError: () => new AutomationDaemonAlreadyRunningError(home),
		});
		handle.release();
		return false;
	} catch (error) {
		if (error instanceof AutomationDaemonAlreadyRunningError) return true;
		return false;
	}
}

const defaultParcelBackend: AutomationWatcherBackend = async (dir, onEvents, onError) => {
	return parcelWatcher.subscribe(
		dir,
		(err, events) => {
			if (err) {
				onError?.(err);
				return;
			}
			onEvents(events.map((ev) => ({ type: ev.type, path: ev.path })));
		},
		{ ignore: PARCEL_IGNORED_SEGMENTS },
	);
};

interface RuleScheduler {
	stop(): Promise<void>;
}

function createRuleScheduler(deps: {
	readonly automation: Automation;
	readonly home: string;
	readonly watcher: AutomationWatcherBackend;
	readonly runner: AutomationRunner;
	readonly maxRunsPerMinute: number;
	readonly onRun?: (automation: Automation, record: AutomationRunRecord) => void;
	readonly onError: (message: string) => void;
}): Promise<RuleScheduler> {
	const { automation, home, watcher, runner, maxRunsPerMinute, onRun, onError } = deps;
	const pending = new Map<string, AutomationEvent>();
	const recentRuns: number[] = [];
	const subscriptions: AutomationSubscription[] = [];
	const homePrefix = `${resolve(home)}/`;
	let timer: NodeJS.Timeout | undefined;
	let inFlight = false;
	let stopped = false;
	let disabled = false;

	const schedule = (): void => {
		if (stopped || disabled || inFlight || pending.size === 0) return;
		clearTimeout(timer);
		timer = setTimeout(fire, automation.debounceMs);
		timer.unref?.();
	};

	const fire = (): void => {
		timer = undefined;
		if (stopped || disabled || inFlight) return;
		const batch = [...pending.values()].filter((ev) => automation.events.includes(ev.type));
		pending.clear();
		if (batch.length === 0) return;

		const now = Date.now();
		while (recentRuns.length > 0 && now - (recentRuns[0] ?? 0) >= RUNAWAY_WINDOW_MS) {
			recentRuns.shift();
		}
		if (recentRuns.length >= maxRunsPerMinute) {
			disabled = true;
			onError(
				`Automation "${automation.id}" disabled after ${maxRunsPerMinute} runs within 60s (possible self-trigger loop)`,
			);
			return;
		}
		recentRuns.push(now);

		inFlight = true;
		void runner(automation, batch)
			.then((record) => {
				if (!stopped) onRun?.(automation, record);
			})
			.catch((err) => {
				onError(`Automation "${automation.id}" failed: ${err instanceof Error ? err.message : String(err)}`);
			})
			.finally(() => {
				inFlight = false;
				schedule();
			});
	};

	const ingest = (root: string, events: readonly AutomationEvent[]): void => {
		if (stopped || disabled) return;
		let added = false;
		for (const ev of events) {
			const abs = resolve(ev.path);
			if (abs === resolve(home) || abs.startsWith(homePrefix)) continue;
			if (!matchesAutomationPath(automation, root, ev.path)) continue;

			const prev = pending.get(ev.path);
			if (prev?.type === "create" && ev.type === "update") {
				// Keep as create: a newly written file flushed in chunks is still a creation.
			} else if (prev?.type === "create" && ev.type === "delete") {
				pending.delete(ev.path);
				continue;
			} else {
				pending.set(ev.path, ev);
			}
			added = true;
		}
		if (added) schedule();
	};

	return (async () => {
		for (const rawRoot of automation.paths) {
			const root = resolve(rawRoot);
			if (!existsSync(root)) {
				onError(`Automation "${automation.id}": watch path does not exist: ${root}`);
				continue;
			}
			try {
				const sub = await watcher(
					root,
					(evs) => ingest(root, evs),
					(err) => onError(`Automation "${automation.id}" watcher error on ${root}: ${err.message}`),
				);
				subscriptions.push(sub);
			} catch (err) {
				onError(
					`Automation "${automation.id}" failed to watch ${root}: ${err instanceof Error ? err.message : String(err)}`,
				);
			}
		}

		return {
			async stop(): Promise<void> {
				stopped = true;
				clearTimeout(timer);
				timer = undefined;
				pending.clear();
				await Promise.allSettled(subscriptions.splice(0).map((s) => s.unsubscribe()));
			},
		};
	})();
}

export async function startAutomationDaemon(options: AutomationDaemonOptions = {}): Promise<AutomationDaemonHandle> {
	const env = options.env ?? process.env;
	const home = resolve(options.home ?? resolveAutoRAGHome(env));
	const configPath = join(home, "automations.json");
	const lockPath = automationDaemonLockPath(home);
	mkdirSync(join(home, "automations"), { recursive: true });

	const lock: FileLockHandle = acquireFileLock(lockPath, {
		timeoutMs: 0,
		staleMs: 0,
		timeoutError: () => new AutomationDaemonAlreadyRunningError(home),
	});

	const watcher = options.watcher ?? defaultParcelBackend;
	const runner: AutomationRunner =
		options.runner ??
		((automation, events) => runAutomation(automation, events, { home, env, selfCommand: options.selfCommand }));
	const maxRunsPerMinute = options.maxRunsPerMinute ?? DEFAULT_MAX_RUNS_PER_MINUTE;
	const onError = options.onError ?? ((msg: string) => process.stderr.write(`[automate] ${msg}\n`));

	let schedulers: RuleScheduler[] = [];
	let stopped = false;

	const startRules = async (automations: readonly Automation[]): Promise<RuleScheduler[]> => {
		const built: RuleScheduler[] = [];
		for (const automation of automations) {
			built.push(
				await createRuleScheduler({
					automation,
					home,
					watcher,
					runner,
					maxRunsPerMinute,
					onRun: options.onRun,
					onError,
				}),
			);
		}
		return built;
	};

	const reload = async (): Promise<void> => {
		if (stopped) return;
		let automations: Automation[];
		try {
			automations = loadAutomations(configPath, env);
		} catch (error) {
			onError(error instanceof Error ? error.message : String(error));
			return;
		}
		const previous = schedulers.splice(0);
		await Promise.allSettled(previous.map((s) => s.stop()));
		if (stopped) return;
		schedulers = await startRules(automations);
	};

	try {
		await reload();
	} catch (error) {
		lock.release();
		throw error;
	}

	const watchingConfig = options.watchConfigFile !== false;
	if (watchingConfig) {
		watchFile(configPath, { interval: CONFIG_POLL_INTERVAL_MS, persistent: false }, (curr, prev) => {
			if (curr.mtimeMs !== prev.mtimeMs || curr.size !== prev.size) {
				void reload();
			}
		});
	}

	return {
		reload,
		async stop(): Promise<void> {
			if (stopped) return;
			stopped = true;
			if (watchingConfig) unwatchFile(configPath);
			const active = schedulers.splice(0);
			await Promise.allSettled(active.map((s) => s.stop()));
			lock.release();
		},
	};
}
