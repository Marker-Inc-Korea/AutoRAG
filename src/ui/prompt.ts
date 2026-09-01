/**
 * Short copy-paste prompt for a coding agent.
 *
 * The operator picks a type, names it, and may fill optional extras
 * (repos, remotes, mailbox). Blank extras become questions the agent
 * must ask the user.
 */

import { ConfigError } from "../cli/config.ts";
import { getDatasourceType, getPickerEntry, type PickerExtra } from "./catalog.ts";

export interface RegistrationPromptInput {
	readonly type: string;
	readonly alias?: string;
	readonly note?: string;
	readonly extras?: Readonly<Record<string, string>>;
}

export interface RegistrationPrompt {
	readonly type: string;
	readonly title: string;
	readonly alias: string;
	readonly prompt: string;
	readonly questions: readonly string[];
}

export function buildRegistrationPrompt(input: RegistrationPromptInput): RegistrationPrompt {
	const entry = getPickerEntry(input.type);
	if (entry === undefined) throw new ConfigError(`Unknown source type: ${input.type}`);
	const alias = (input.alias?.trim() || entry.type).trim();
	const note = input.note?.trim();
	const skill = getDatasourceType(entry.type);
	const extras = input.extras ?? {};
	const chosen: string[] = [];
	const questions: string[] = [];
	for (const extra of entry.extras) {
		const value = filledExtra(extras, extra);
		if (value !== undefined) chosen.push(`${extra.label}: ${value}`);
		else questions.push(extra.question);
	}

	const lines = [
		`Register this AutoRAG datasource on this computer (trusted config.json, not librarian tools).`,
		"",
		`Type: ${entry.title} (\`${entry.type}\`)`,
		`Name: ${alias}`,
		`What's in it: ${note && note.length > 0 ? note : "(not specified — ask me if you need more context)"}`,
	];
	if (!entry.supportsMultiple) {
		lines.push("KakaoTalk is single-account. Do not ask which Kakao account to add.");
	}
	if (chosen.length > 0) {
		lines.push("", "Already chosen:", ...chosen.map((item) => `- ${item}`));
	}
	if (questions.length > 0) {
		lines.push("", "Ask me these before writing config:", ...questions.map((item) => `- ${item}`));
	}
	lines.push("");
	if (entry.binaryName) {
		lines.push(
			`Find or install the \`${entry.binaryName}\` CLI on PATH${entry.installHint ? ` (${entry.installHint})` : ""}. Do not ask me for a CLI path unless it is missing.`,
		);
	}
	lines.push(
		`Add \`datasources.${alias}\` with \`type: "${entry.type}"\`, enabled true.`,
		`Grant access with tags ${JSON.stringify(skill?.defaultTags ?? [entry.type])} and scope \`/${alias}/**\`.`,
		"Never ask this UI for a token, password, cookie, secret, or refresh credential.",
		"If authentication is needed, ask me to set the secret in the local environment, or in the CLI's own keychain/profile, then store only the environment-variable name in AutoRAG config.",
		"Do not print, echo, log, or copy secret values. Verify only that the referenced local secret is available.",
	);
	return { type: entry.type, title: entry.title, alias, prompt: lines.join("\n"), questions };
}

function filledExtra(extras: Readonly<Record<string, string>>, extra: PickerExtra): string | undefined {
	const raw = extras[extra.key]?.trim() ?? "";
	if (raw.length === 0 || raw === "other") return extras[`${extra.key}Other`]?.trim() || undefined;
	return raw;
}
