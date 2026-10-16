import { parentPort } from "node:worker_threads";

/**
 * Stand-in parse worker that misbehaves by file name: it spins, exhausts its
 * heap, throws outside any handler, or exits. Speaks the same message protocol
 * as src/parser/parse-worker.ts.
 */
interface Request {
	readonly id: number;
	readonly virtualPath: string;
}

parentPort?.on("message", (request: Request) => {
	const { id, virtualPath } = request;
	if (virtualPath.endsWith("/spin.bin")) {
		for (;;) {
			// CPU-bound synchronous loop: no timer or promise can interrupt it.
		}
	}
	if (virtualPath.endsWith("/oom.bin")) {
		const hoard: unknown[] = [];
		for (;;) hoard.push(new Array(100_000).fill(Math.random()));
	}
	if (virtualPath.endsWith("/crash.bin")) {
		setImmediate(() => {
			throw new Error("worker exploded: crash sentinel");
		});
		return;
	}
	if (virtualPath.endsWith("/exit.bin")) {
		process.exit(3);
	}
	if (virtualPath.endsWith("/rooted.bin")) {
		parentPort?.postMessage({
			id,
			ok: false,
			kind: "parse-error",
			parserName: "kordoc",
			message: "root cause sentinel",
			stack: "Error: root cause sentinel\n    at fixture",
		});
		return;
	}
	parentPort?.postMessage({ id, ok: true, output: { markdown: `fine ${virtualPath}` } });
});
