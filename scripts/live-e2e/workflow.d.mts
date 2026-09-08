export function isFingerprintCurrent(previous: Readonly<Record<string, unknown>>, current: Readonly<Record<string, unknown>>): boolean;
export function assertServiceReady(result: Readonly<{ verdict: string; code?: string }>): void;
import type { AutoRAGAgentOptions } from "../../src/agent/agent.ts";

export function buildLiveStackOptions(root: string, workspace: string): AutoRAGAgentOptions;
export function assertAbsoluteReadableSource(source: string): string;
export function tryAcquireWorkflowLock(root?: string): { readonly ok: boolean; readonly code?: string; readonly release?: () => void };
export function cleanupCloneState(clonePath: string, sharedRoot: string): void;
export function runWorkflow(options: { readonly root: string; readonly mode: "cold" | "warm"; readonly evidenceDir?: string }): Promise<Readonly<Record<string, unknown>>>;
export const REPO_ROOT: string;
