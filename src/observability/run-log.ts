import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

export type AutoRAGRunEvent =
	| {
			event: "search_started";
			timestamp: string;
			sessionId: string;
			queryLength: number;
			orchestratorModel: string;
			explorerModel: string;
	  }
	| {
			event: "search_completed";
			timestamp: string;
			sessionId: string;
			resultCount: number;
	  }
	| {
			event: "search_failed";
			timestamp: string;
			sessionId: string;
			errorType: string;
	  }
	| {
			event: "cleanup_failed";
			timestamp: string;
			sessionId: string;
			failureCount: number;
			errorTypes: string[];
	  }
	| {
			event: "dispatch_rejected";
			schemaVersion: 1;
			sessionId: string;
			toolCallId: string | null;
			sequence: number;
			timestamp: string;
			dispatchKind: "launch" | "admin" | "control" | "mutation" | "schedule" | "hybrid" | "unknown";
			code: string;
			field: string;
			forceCorrectable: boolean;
	  }
	| {
			event: "dispatch_autofilled";
			schemaVersion: 1;
			sessionId: string;
			toolCallId: string | null;
			sequence: number;
			timestamp: string;
			dispatchKind: "launch";
			fields: {
				artifacts: boolean;
				agentScope: boolean;
				leafModelFillCount: number;
			};
	  };

export class AutoRAGRunLogger {
	private readonly path: string;

	constructor(path: string) {
		this.path = path;
	}

	write(event: AutoRAGRunEvent): void {
		try {
			mkdirSync(dirname(this.path), { recursive: true });
			appendFileSync(this.path, `${JSON.stringify(event)}\n`, "utf8");
		} catch (error) {
			// Run logging is diagnostic only; never change search behavior for logger failures.
			void error;
		}
	}
}
