/**
 * Secret and one-time-code redaction, applied wherever text leaves the user's machine for a model
 * (agent context, decision/judgment requests) or lands in an index (mirror markdown, connector chunks).
 *
 * The vendor-token patterns are ported from Gitleaks (https://github.com/gitleaks/gitleaks, MIT,
 * Copyright (c) 2019 Zachary Rice; see licenses/gitleaks-MIT.txt). Gitleaks' per-rule `keywords`
 * pre-filter is kept: a rule's regex only runs when a cheap `includes` finds one of its keywords, so
 * ordinary text costs a handful of substring scans. Every pattern is linear: bounded quantifiers, no
 * nested repetition, no unbounded lazy scans.
 *
 * Redaction is idempotent and never echoes a secret; it replaces the secret part with
 * `[REDACTED:<rule>]` and keeps the surrounding text (key names, `Bearer`, URL host).
 */

/** Rule ids appear in the replacement marker and in reports. */
export type RedactionFindings = Record<string, number>;

export interface RedactionResult {
	readonly text: string;
	/** Count of masked secrets per rule id. Never contains secret material. */
	readonly findings: RedactionFindings;
}

interface Rule {
	readonly id: string;
	/** Case-sensitive substrings; the rule is skipped unless one occurs. */
	readonly keywords?: readonly string[];
	/** Lowercase substrings checked against the lowercased text. */
	readonly keywordsLower?: readonly string[];
	/** Group `sec` is replaced when present, else the whole match. */
	readonly regex: RegExp;
	/** Return false to keep a candidate (placeholder, identifier, low-entropy value). */
	readonly accept?: (secret: string) => boolean;
}

const TERMINATOR = String.raw`(?![\w-])`;

function entropy(value: string): number {
	const counts = new Map<string, number>();
	for (const char of value) counts.set(char, (counts.get(char) ?? 0) + 1);
	let bits = 0;
	for (const count of counts.values()) {
		const p = count / value.length;
		bits -= p * Math.log2(p);
	}
	return bits;
}

const PLACEHOLDER = /^(?:[*#x•._-]+|\$.*|\{\{.*|<.*|%.*|\[REDACTED.*)$/iu;
const PLACEHOLDER_WORDS = /your|example|placeholder|changeme|process\.env|os\.environ|getenv|env\./iu;

function isPlaceholder(value: string): boolean {
	return PLACEHOLDER.test(value) || PLACEHOLDER_WORDS.test(value) || /[()]/u.test(value);
}

/** A key=value credential: not a placeholder, and shaped like a secret rather than an identifier. */
function looksLikeCredential(value: string): boolean {
	if (isPlaceholder(value)) return false;
	const mixed = /\d/u.test(value) || /[^\w]/u.test(value);
	const bits = entropy(value);
	return (mixed && bits >= 2.5) || (value.length >= 24 && bits >= 3.5);
}

const PASSWORD_VALUE = String.raw`[^\s"'\x60,;]`;

const OTP_WORDS = [
	"otp",
	"totp",
	"2fa",
	"mfa",
	"passcode",
	"one[- ]?time (?:pass(?:word|code)|code|pin)",
	"(?:verification|security|authentication|authorization|confirmation|login|sign-?in|auth) (?:code|pin)",
	String.raw`\d[- ]digit (?:code|pin|otp)`,
	String.raw`your (?:\w+ ){0,2}code`,
	String.raw`인증\s?(?:번호|코드)`,
	String.raw`보안\s?(?:번호|코드)`,
	String.raw`확인\s?(?:번호|코드)`,
	String.raw`일회용\s?(?:비밀번호|번호|코드)`,
	String.raw`승인\s?번호`,
].join("|");
const OTP_GAP = String.raw`(?:[\s:：=\-–\[("'은는이가]|\b(?:is|are|was)\b){0,10}`;

const RULES: readonly Rule[] = [
	{
		id: "private-key",
		keywords: ["-----BEGIN"],
		regex: /-----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----[A-Za-z0-9+/=\s]{16,16384}(?:-----END[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----)?/u,
	},
	{
		id: "anthropic-api-key",
		keywords: ["sk-ant-"],
		regex: new RegExp(String.raw`\bsk-ant-[a-z0-9]{2,6}-[A-Za-z0-9_-]{40,200}${TERMINATOR}`, "u"),
	},
	{
		id: "openrouter-api-key",
		keywords: ["sk-or-"],
		regex: new RegExp(String.raw`\bsk-or-v1-[a-f0-9]{48,80}${TERMINATOR}`, "u"),
	},
	{
		id: "openai-api-key",
		keywords: ["sk-"],
		regex: new RegExp(
			String.raw`\bsk-(?:(?:proj|svcacct|admin)-[A-Za-z0-9_-]{40,200}|[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}|[A-Za-z0-9]{40,})${TERMINATOR}`,
			"u",
		),
	},
	{
		id: "aws-access-token",
		keywords: ["AKIA", "ASIA", "ABIA", "ACCA", "A3T"],
		regex: /\b(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b/u,
	},
	{
		id: "github-token",
		keywords: ["gh", "github_pat_"],
		regex: new RegExp(String.raw`\b(?:gh[pousr]_[0-9a-zA-Z]{36}|github_pat_\w{82})${TERMINATOR}`, "u"),
	},
	{
		id: "gitlab-token",
		keywords: ["glpat-"],
		regex: new RegExp(String.raw`\bglpat-[\w-]{20}${TERMINATOR}`, "u"),
	},
	{
		id: "slack-token",
		keywords: ["xox"],
		regex: /\bxox[abeprs]-[0-9a-zA-Z-]{10,200}/u,
	},
	{
		id: "slack-webhook",
		keywords: ["hooks.slack.com"],
		regex: /hooks\.slack\.com\/(?:services|workflows|triggers)\/(?<sec>[A-Za-z0-9+/_-]{20,100})/u,
	},
	{
		id: "stripe-access-token",
		keywords: ["sk_", "rk_"],
		regex: new RegExp(String.raw`\b[sr]k_(?:test|live|prod)_[a-zA-Z0-9]{10,99}${TERMINATOR}`, "u"),
	},
	{
		id: "gcp-api-key",
		keywords: ["AIza"],
		regex: new RegExp(String.raw`\bAIza[\w-]{35}${TERMINATOR}`, "u"),
	},
	{
		id: "google-oauth-token",
		keywords: ["ya29."],
		regex: new RegExp(String.raw`\bya29\.[\w-]{20,300}${TERMINATOR}`, "u"),
	},
	{
		id: "huggingface-access-token",
		keywords: ["hf_"],
		regex: new RegExp(String.raw`\bhf_[A-Za-z]{34}${TERMINATOR}`, "u"),
	},
	{
		id: "xai-api-key",
		keywords: ["xai-"],
		regex: new RegExp(String.raw`\bxai-[A-Za-z0-9]{60,100}${TERMINATOR}`, "u"),
	},
	{
		id: "groq-api-key",
		keywords: ["gsk_"],
		regex: new RegExp(String.raw`\bgsk_[A-Za-z0-9]{40,60}${TERMINATOR}`, "u"),
	},
	{
		id: "npm-access-token",
		keywords: ["npm_"],
		regex: new RegExp(String.raw`\bnpm_[A-Za-z0-9]{36}${TERMINATOR}`, "u"),
	},
	{
		id: "sendgrid-api-token",
		keywords: ["SG."],
		regex: new RegExp(String.raw`\bSG\.[A-Za-z0-9=_.-]{66}${TERMINATOR}`, "u"),
	},
	{
		id: "telegram-bot-token",
		keywords: [":AA"],
		regex: new RegExp(String.raw`\b\d{8,10}:AA[A-Za-z0-9_-]{33}${TERMINATOR}`, "u"),
	},
	{
		id: "discord-bot-token",
		keywords: ["."],
		regex: new RegExp(String.raw`\b[MNO][A-Za-z\d]{23,25}\.[\w-]{6}\.[\w-]{27,38}${TERMINATOR}`, "u"),
		accept: (value) => entropy(value) >= 3.5 && /\d/u.test(value),
	},
	{
		id: "jwt",
		keywords: ["ey"],
		regex: /\bey[A-Za-z0-9]{17,}\.ey[A-Za-z0-9/\\_-]{17,}\.(?:[A-Za-z0-9/\\_-]{10,}={0,2})?/u,
	},
	{
		id: "totp-secret",
		keywordsLower: ["otpauth"],
		regex: /otpauth(?:-migration)?:\/\/[^\s"'<>]{0,300}?[?&](?:secret|data)=(?<sec>[A-Za-z0-9%+/=_-]{8,500})/iu,
	},
	{
		id: "bearer-token",
		keywordsLower: ["bearer"],
		regex: /\bbearer\s+(?<sec>[A-Za-z0-9._~+/=-]{20,2000})/iu,
		accept: (value) => !isPlaceholder(value),
	},
	{
		id: "url-credentials",
		keywords: ["://"],
		regex: /\b[a-z][a-z0-9+.-]{1,20}:\/\/[^\s:/@]{1,100}:(?<sec>[^\s/@]{3,200})@/iu,
		accept: (value) => !isPlaceholder(value),
	},
	{
		id: "otp-code",
		keywordsLower: [
			"otp",
			"2fa",
			"mfa",
			"passcode",
			"one-time",
			"one time",
			"onetime",
			"code",
			"pin",
			"sign in",
			"sign-in",
			"log in",
			"log-in",
			"verify",
			"authenticate",
			"인증",
			"보안",
			"확인",
			"일회용",
			"승인",
		],
		regex: new RegExp(
			String.raw`(?:${OTP_WORDS})${OTP_GAP}(?<sec>\d{4,8})(?!\d)|(?<![\w-])(?<sec2>\d{4,8})(?![\w-])(?=\s+(?:(?:is|as)\s+(?:your|the)\s+(?:\w+\s+){0,3}(?:code|otp|passcode|pin)|to\s+(?:sign[- ]?in|log[- ]?in|verify|authenticate|confirm))\b)`,
			"iu",
		),
	},
	{
		id: "password",
		keywordsLower: ["password", "passwd", "pwd", "비밀번호", "패스워드", "비번"],
		regex: new RegExp(
			String.raw`(?:password|passwd|pwd|비밀번호|패스워드|비번)[\w.-]{0,40}["']?\s*(?:[:：=]|=>|은|는)\s*["']?(?<sec>${PASSWORD_VALUE}{6,200})`,
			"iu",
		),
		accept: looksLikeCredential,
	},
	{
		id: "generic-secret",
		keywordsLower: [
			"secret",
			"token",
			"api_key",
			"api-key",
			"apikey",
			"access_key",
			"access-key",
			"credential",
			"private_key",
		],
		regex: new RegExp(
			String.raw`(?:secret|token|api[_-]?key|access[_-]?key|private[_-]?key|credential)[\w.-]{0,40}["']?\s*(?:[:：=]|=>)\s*["']?(?<sec>${PASSWORD_VALUE}{8,200})`,
			"iu",
		),
		accept: looksLikeCredential,
	},
];

/** Disables redaction when set to `0`, `false`, or `off`; for debugging only. */
export function secretRedactionEnabled(env: Readonly<Record<string, string | undefined>> = process.env): boolean {
	const value = env.AUTORAG_REDACT_SECRETS?.trim().toLowerCase();
	return !(value === "0" || value === "false" || value === "off");
}

const GLOBAL_FLAGS = (flags: string) => (flags.includes("g") ? flags : `${flags}g`);
const globalRegex = new Map<Rule, RegExp>();
function matcherFor(rule: Rule): RegExp {
	let regex = globalRegex.get(rule);
	if (regex === undefined) {
		regex = new RegExp(rule.regex.source, GLOBAL_FLAGS(rule.regex.flags));
		globalRegex.set(rule, regex);
	}
	return regex;
}

function applies(rule: Rule, text: string, lower: () => string): boolean {
	if (rule.keywords === undefined && rule.keywordsLower === undefined) return true;
	if (rule.keywords?.some((keyword) => text.includes(keyword))) return true;
	if (rule.keywordsLower !== undefined) {
		const haystack = lower();
		return rule.keywordsLower.some((keyword) => haystack.includes(keyword));
	}
	return false;
}

const MIN_LENGTH = 6;

/** Masks secrets in `text` and reports what was found. Returns the same string when nothing matched. */
export function redactSecretsWithReport(text: string): RedactionResult {
	if (text.length < MIN_LENGTH) return { text, findings: {} };
	let current = text;
	let findings: RedactionFindings | undefined;
	let lowered: string | undefined;
	let loweredFor: string | undefined;
	const lower = (): string => {
		if (loweredFor !== current) {
			lowered = current.toLowerCase();
			loweredFor = current;
		}
		return lowered as string;
	};
	for (const rule of RULES) {
		if (!applies(rule, current, lower)) continue;
		const marker = `[REDACTED:${rule.id}]`;
		let count = 0;
		const replaced = current.replace(matcherFor(rule), (...args) => {
			const groups = args[args.length - 1] as Record<string, string | undefined> | string;
			const named = typeof groups === "object" ? groups : undefined;
			const match = args[0] as string;
			const secret = named?.sec ?? named?.sec2;
			if (secret === undefined) {
				if (rule.accept !== undefined && !rule.accept(match)) return match;
				count += 1;
				return marker;
			}
			if (rule.accept !== undefined && !rule.accept(secret)) return match;
			count += 1;
			const at = match.lastIndexOf(secret);
			return `${match.slice(0, at)}${marker}${match.slice(at + secret.length)}`;
		});
		if (count > 0) {
			current = replaced;
			findings ??= {};
			findings[rule.id] = (findings[rule.id] ?? 0) + count;
		}
	}
	return { text: current, findings: findings ?? {} };
}

/** Masks secrets in `text`. Returns the same string when nothing matched. */
export function redactSecrets(text: string): string {
	return redactSecretsWithReport(text).text;
}
