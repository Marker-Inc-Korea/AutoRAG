/**
 * `@autorag/librarian/profiles`: agent profiles for helper features.
 *
 * A profile is a system prompt, a tool set, and a thinking level; the factory turns
 * one into a plain in-memory pi session on the AutoRAG agent's own model and
 * credentials. The desktop app's smart indexing and file-triggered automations
 * are built on it. No profile session touches AutoRAG's retrieval memory.
 */
export {
	type AgentProfile,
	createProfileSession,
	type ProfileContext,
	type ProfileModel,
	type ProfileSessionOptions,
	type ProfileThinking,
} from "./profiles/agent-factory.ts";
export {
	type AutomationProfileDefinition,
	type AutomationProfileInput,
	getAutomationProfile,
	listAutomationProfiles,
	registerAutomationProfile,
} from "./profiles/automation-profiles.ts";
export { includedExtensions, type LoopGuardRule, loopGuardPrompt } from "./profiles/loop-guard.ts";
export {
	createRetrievalToolkit,
	hasRetrievalTools,
	type RetrievalToolkit,
	type RetrievalToolOptions,
} from "./profiles/retrieval-tools.ts";
export {
	applyDazziThinkingLevels,
	openPlanRuntime,
	type PlanRuntime,
	preferPlanOAuth,
	resolveConfiguredSessionModel,
	resolveSessionModel,
	type SessionModelAuth,
} from "./profiles/session-model.ts";
