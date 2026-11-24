import { describe, expect, it } from "vitest";
import { redactSecrets, redactSecretsWithReport } from "../../src/privacy/redact.ts";

// Fixtures are assembled at runtime so no literal credential-shaped string sits in the repository.
const rep = (chunk: string, n: number) => chunk.repeat(n);
const AWS = `AKIA${rep("ABCD", 4)}`;
const GH = `ghp_${rep("aB3dE6gH9k", 3)}abcdef`;
const SLACK = `xoxb-${rep("1234567890", 1)}-${rep("0987654321", 1)}-abcDEF123`;
const OPENAI_PROJ = `sk-proj-${rep("Ab1_", 20)}`;
const ANTHROPIC = `sk-ant-api03-${rep("Zy9-", 25)}AA`;
const STRIPE = `sk_${"live"}_${rep("aZ09", 6)}`;
const GOOGLE = `AIza${rep("Sy1-", 8)}abc`;
const HF = `hf_${rep("abcdefghij", 3)}abcd`;
const JWT = `eyJ${rep("hbGciOiJI", 3)}.eyJ${rep("zdWIiOiIx", 3)}.${rep("SflKxwRJSM", 2)}`;
const PEM = [
	`${"-----"}BEGIN RSA PRIVATE KEY${"-----"}`,
	rep("MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn", 6),
	`${"-----"}END RSA PRIVATE KEY${"-----"}`,
].join("\n");

describe("redactSecrets: vendor tokens", () => {
	const cases: Array<[string, string, string]> = [
		["aws-access-token", AWS, `key ${AWS} end`],
		["github-token", GH, `export X=${GH}`],
		["slack-token", SLACK, `token: ${SLACK}`],
		["openai-api-key", OPENAI_PROJ, `OPENAI ${OPENAI_PROJ}!`],
		["anthropic-api-key", ANTHROPIC, `${ANTHROPIC}`],
		["stripe-access-token", STRIPE, `"${STRIPE}"`],
		["gcp-api-key", GOOGLE, `url?key=${GOOGLE}&x=1`],
		["huggingface-access-token", HF, `HF ${HF}`],
		["jwt", JWT, `Cookie session=${JWT};`],
		["private-key", PEM, `before\n${PEM}\nafter`],
	];
	for (const [rule, secret, text] of cases) {
		it(`masks ${rule}`, () => {
			const out = redactSecrets(text);
			expect(out).not.toContain(secret);
			expect(out).toContain(`[REDACTED:${rule}]`);
		});
	}

	it("keeps surrounding text intact", () => {
		expect(redactSecrets(`a ${AWS} b`)).toBe("a [REDACTED:aws-access-token] b");
	});
});

describe("redactSecrets: OTP and credentials in context", () => {
	it.each([
		["Your verification code is 482913. Do not share.", "482913"],
		["OTP: 7391", "7391"],
		["[Web발신] 인증번호 [558201] 를 입력해주세요", "558201"],
		["인증번호는 123456입니다", "123456"],
		["one-time passcode 90817263", "90817263"],
		["123456 is your Google verification code", "123456"],
		["Use 6-digit code 654321 to log in (2FA)", "654321"],
		["Use code 884210 to sign in", "884210"],
		["Enter 904122 to verify your account", "904122"],
	])("masks OTP in %j", (text, code) => {
		const out = redactSecrets(text);
		expect(out).not.toContain(code);
		expect(out).toContain("[REDACTED:otp-code]");
	});

	it("masks totp secret in otpauth URI", () => {
		const seed = ["JBSWY3DP", "EHPK3PXP"].join("");
		const out = redactSecrets(`otpauth://totp/me?secret=${seed}&issuer=x`);
		expect(out).not.toContain(seed);
		expect(out).toContain("issuer=x");
	});

	it("masks bearer tokens and keeps the scheme", () => {
		const out = redactSecrets(`Authorization: Bearer ${rep("abcDEF123456", 3)}`);
		expect(out).toContain("Bearer [REDACTED:bearer-token]");
	});

	it("masks password in URL credentials", () => {
		const pw = ["Sup3r", "S3cretPw"].join("");
		const out = redactSecrets(`postgres://admin:${pw}@db.local:5432/app`);
		expect(out).toBe("postgres://admin:[REDACTED:url-credentials]@db.local:5432/app");
	});

	it("masks generic high-entropy assignments but keeps the key name", () => {
		const val = ["q8Zr2LmW", "0xYt5VbN", "7cKd3HsP", "9aJe1UgF"].join("");
		const out = redactSecrets(`client_secret = "${val}"`);
		expect(out).toContain("client_secret");
		expect(out).not.toContain(val);
	});

	it("masks Korean password assignments", () => {
		const out = redactSecrets("비밀번호: hunter2pass!");
		expect(out).not.toContain("hunter2pass");
	});
});

describe("redactSecrets: no false positives on ordinary text", () => {
	it.each([
		"The meeting is at 10:30 on 2026-11-24 in room 4021.",
		"Order 1234567 shipped; call 010-1234-5678.",
		"api_key = process.env.OPENAI_API_KEY",
		`token: "$\u007bGITHUB_TOKEN}"`,
		'API_KEY="your_api_key_here_please_replace"',
		"git commit 55fb3cc3b0c1d2e3f405162738495a6b7c8d9e0f",
		"password: ********",
		"sk-learn is not a key; neither is task-1234567890",
		"We discuss one-time pad encryption and code review.",
		"한국어 문서에는 인증 절차가 설명되어 있다. 코드는 3개다.",
	])("leaves %j unchanged", (text) => {
		expect(redactSecrets(text)).toBe(text);
	});
});

describe("redactSecretsWithReport", () => {
	it("returns rule ids and counts without echoing secrets", () => {
		const { text, findings } = redactSecretsWithReport(`${AWS} ${AWS} ${GH}`);
		expect(findings).toEqual({ "aws-access-token": 2, "github-token": 1 });
		expect(JSON.stringify(findings)).not.toContain(AWS);
		expect(text).not.toContain(GH);
	});

	it("is idempotent", () => {
		const once = redactSecrets(`${AWS} OTP: 482913`);
		expect(redactSecrets(once)).toBe(once);
	});

	it("returns the same string instance when nothing matches", () => {
		const input = "plain text ".repeat(100);
		expect(redactSecrets(input)).toBe(input);
	});
});

describe("redactSecrets: latency", () => {
	it("scans a 1 MB clean document in well under 50 ms (median)", () => {
		const doc =
			"The quick brown fox jumps over the lazy dog. 한국어 문장입니다. token budget and password policy notes.\n".repeat(
				10_000,
			);
		redactSecrets(doc);
		const runs: number[] = [];
		for (let i = 0; i < 5; i++) {
			const t = performance.now();
			redactSecrets(doc);
			runs.push(performance.now() - t);
		}
		runs.sort((a, b) => a - b);
		expect(runs[2]).toBeLessThan(50);
	});

	it("does not blow up on adversarial repetition", () => {
		const t = performance.now();
		redactSecrets(`password=${"a".repeat(200_000)}`);
		redactSecrets(`code ${"1".repeat(200_000)}`);
		redactSecrets("-----BEGIN PRIVATE KEY-----".repeat(5_000));
		expect(performance.now() - t).toBeLessThan(500);
	});
});
