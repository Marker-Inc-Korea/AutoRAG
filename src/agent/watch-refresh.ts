/**
 * Filesystem watch scheduler for incremental refresh. Deliberately dependency-
 * injected (via {@link WatcherFactory}) so it is unit-testable without real
 * filesystem timing, and so the agent can pass a real `fs.watch`-backed factory.
 *
 * Invariants:
 *  - Debounced: a burst of change events collapses into one refresh.
 *  - Backpressure: at most one refresh in flight plus at most one pending rerun.
 *  - Stoppable: after stop() no further refresh is scheduled and every watcher
 *    is closed; late events are ignored.
 *  - Bounded: never creates more than `maxWatchers` watchers.
 */

export interface WatchWatcher {
	close(): void;
}

export type WatcherFactory = (dir: string, onChange: (filename: string | null) => void) => WatchWatcher;

export interface WatchRefreshDeps {
	readonly dirs: readonly string[];
	readonly debounceMs: number;
	readonly maxWatchers: number;
	readonly watcherFactory: WatcherFactory;
	readonly runRefresh: () => Promise<void>;
	/** Called once when the number of directories exceeds `maxWatchers`. */
	readonly onLimit?: () => void;
}

export interface WatchRefreshHandle {
	stop(): void;
}

const EXCLUDED_SEGMENTS = new Set([".autorag", ".git", "node_modules"]);

function isExcluded(filename: string | null): boolean {
	if (!filename) return false;
	return filename.split(/[/\\]/).some((segment) => EXCLUDED_SEGMENTS.has(segment));
}

export function createWatchRefresh(deps: WatchRefreshDeps): WatchRefreshHandle {
	let stopped = false;
	let inFlight = false;
	let pending = false;
	let timer: ReturnType<typeof setTimeout> | undefined;
	const watchers: WatchWatcher[] = [];

	const fire = (): void => {
		timer = undefined;
		if (stopped) return;
		if (inFlight) {
			pending = true;
			return;
		}
		inFlight = true;
		void deps
			.runRefresh()
			.catch(() => {
				// Refresh failures are captured by the refresh-status layer, not here.
			})
			.finally(() => {
				inFlight = false;
				if (pending && !stopped) {
					pending = false;
					schedule();
				}
			});
	};

	const schedule = (): void => {
		if (stopped) return;
		if (timer !== undefined) clearTimeout(timer);
		timer = setTimeout(fire, deps.debounceMs);
		timer.unref?.();
	};

	const onChange = (filename: string | null): void => {
		if (stopped || isExcluded(filename)) return;
		schedule();
	};

	const dirsToWatch = deps.dirs.slice(0, deps.maxWatchers);
	if (deps.dirs.length > deps.maxWatchers) deps.onLimit?.();
	for (const dir of dirsToWatch) {
		watchers.push(deps.watcherFactory(dir, onChange));
	}

	return {
		stop(): void {
			stopped = true;
			if (timer !== undefined) {
				clearTimeout(timer);
				timer = undefined;
			}
			pending = false;
			for (const watcher of watchers) watcher.close();
		},
	};
}
