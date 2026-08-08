import { detect } from "chardet";
import iconv from "iconv-lite";
import { describe, expect, it, vi } from "vitest";
import { decodeText } from "../../src/parser/text.ts";

/**
 * Equivalence safety net for the decodeText fast path.
 *
 * `decodeText` produces the text stored in the mirror, so the fast path (skip chardet for
 * pure-ASCII bytes) is only legal if it reproduces the pre-change implementation byte for byte.
 * The pre-change implementation is copied below verbatim (git HEAD of src/parser/text.ts) and
 * every test compares the two as black boxes, including their failure behavior: an input that
 * made the old implementation throw must still throw.
 *
 * chardet is mocked at module level so one test can prove the fast path really skips `detect()`
 * (a throwing detect leaves pure-ASCII decodeText unharmed) while every other test runs the real
 * detector through the same indirection.
 */

const detectImpl = vi.hoisted(() => ({
	current: null as null | ((bytes: Uint8Array) => string | null),
}));

vi.mock("chardet", async (importOriginal) => {
	const actual = await importOriginal<typeof import("chardet")>();
	return {
		...actual,
		detect: (bytes: Uint8Array) => (detectImpl.current ?? actual.detect)(bytes),
	};
});

const WINDOWS_949_LABELS = new Set(["EUC-KR", "ISO-2022-KR", "windows-949", "CP949"]);

/** Verbatim copy of the pre-fast-path decodeText from git HEAD of src/parser/text.ts. */
function referenceDecodeText(bytes: Uint8Array): string {
	const buffer = Buffer.from(bytes);
	const utf8 = iconv.decode(buffer, "utf8");
	if (referenceReplacementCount(utf8) > 0) {
		const cp949 = iconv.decode(buffer, "cp949");
		if (referenceReplacementCount(cp949) < referenceReplacementCount(utf8)) return cp949.normalize("NFC");
	}
	const detected = detect(buffer);
	const encoding = detected && WINDOWS_949_LABELS.has(detected) ? "cp949" : (detected ?? "utf8");
	return iconv.decode(buffer, encoding).normalize("NFC");
}

function referenceReplacementCount(value: string): number {
	return [...value].filter((character) => character === "\uFFFD").length;
}

type Outcome = { readonly kind: "ok"; readonly value: string } | { readonly kind: "throw"; readonly message: string };

function outcomeOf(decode: (bytes: Uint8Array) => string, bytes: Uint8Array): Outcome {
	try {
		return { kind: "ok", value: decode(bytes) };
	} catch (error) {
		return { kind: "throw", message: error instanceof Error ? error.message : String(error) };
	}
}

function assertSameOutcome(bytes: Uint8Array): void {
	expect(outcomeOf(decodeText, bytes)).toEqual(outcomeOf(referenceDecodeText, bytes));
}

describe("decodeText fast path equivalence", () => {
	it("keeps pure ASCII byte-identical to the pre-change implementation", () => {
		const cases = [
			Buffer.from("hello world"),
			Buffer.from("hello world\nthis is a test\n"),
			Buffer.from("# Document 0\n\nRefund approval policy and escalation threshold.\n"),
			Buffer.from("a\tb\tc\r\nd"),
			Buffer.from("0123456789 !@#$%^&*()_+-=[]{}|;':\",./<>?`~"),
			Buffer.from(new Uint8Array(Array.from({ length: 127 }, (_, i) => i + 1))), // every byte 0x01-0x7F
			Buffer.from("ends without newline"),
			Buffer.from("\n\n\n"),
			Buffer.from("line1\r\nline2\r\nline3"),
		];
		for (const bytes of cases) assertSameOutcome(bytes);
	});

	it("keeps ESC-containing ASCII on the legacy path, throw included", () => {
		const cases = [
			Buffer.from("ESC \x1b$B test\n"),
			Buffer.from("\x1b$B"),
			Buffer.from("\x1b$B\x1b(Bplain"),
			Buffer.from("\x1b$C ESC \x1b$B mixed"),
		];
		for (const bytes of cases) assertSameOutcome(bytes);
		// A full ISO-2022-JP escape sequence made the old implementation throw (iconv-lite cannot
		// decode ISO-2022-JP); the fast path must not silently change that into a clean string.
		expect(outcomeOf(referenceDecodeText, Buffer.from("ESC \x1b$B test\n")).kind).toBe("throw");
		expect(outcomeOf(referenceDecodeText, Buffer.from("\x1b$B")).kind).toBe("throw");
		// A 2-byte ESC prefix is too short for the ISO-2022-JP detector; UTF-32LE claims the
		// buffer and iconv drops the incomplete unit, so the legacy output is "".
		expect(outcomeOf(referenceDecodeText, Buffer.from("\x1b$"))).toEqual({ kind: "ok", value: "" });
	});

	it("keeps Korean UTF-8 identical", () => {
		assertSameOutcome(Buffer.from("한글 테스트 문서입니다\n안녕하세요, 세계!\n", "utf8"));
		assertSameOutcome(Buffer.from("가나다라마바사아자차카타파하", "utf8"));
	});

	it("keeps the CP949 fallback path alive", () => {
		// CP949 bytes for "한글" and a longer legacy-encoded paragraph.
		assertSameOutcome(Buffer.from([0xc7, 0xd1, 0xb1, 0xdb]));
		const cp949 = iconv.encode("한글 문서입니다. 오늘은 화요일입니다.", "cp949");
		assertSameOutcome(cp949);
	});

	it("keeps UTF-8 BOM input identical", () => {
		assertSameOutcome(Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from("hello world\n")]));
		assertSameOutcome(Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from("한글 BOM 문서\n", "utf8")]));
		assertSameOutcome(Buffer.from([0xef, 0xbb, 0xbf]));
	});

	it("keeps the empty buffer identical", () => {
		assertSameOutcome(new Uint8Array(0));
	});

	it("keeps invalid byte sequences identical", () => {
		const cases = [
			new Uint8Array([0x80]),
			new Uint8Array([0xc3]), // truncated two-byte sequence
			new Uint8Array([0xe2, 0x82]), // truncated three-byte sequence
			new Uint8Array([0xf0, 0x9f, 0x92]), // truncated four-byte sequence
			new Uint8Array([0xff, 0xff, 0xff]),
			new Uint8Array([0xc3, 0x28]), // invalid continuation
			new Uint8Array([0x61, 0xc3, 0x28]), // valid byte then an invalid sequence
			new Uint8Array([0xfe, 0x80, 0x80, 0x80]),
		];
		for (const bytes of cases) assertSameOutcome(bytes);
	});

	it("keeps NFC-normalizing input identical (combining characters)", () => {
		// Decomposed Hangul and a decomposed Latin letter: NFC must actually fire in both paths.
		assertSameOutcome(Buffer.from("한글", "utf8"));
		assertSameOutcome(Buffer.from("e\u0301", "utf8"));
		assertSameOutcome(Buffer.from("cafe\u0301 \u1100\u1161", "utf8"));
	});

	it("keeps null-byte input identical", () => {
		const cases = [
			new Uint8Array([0x61, 0x00, 0x62]),
			new Uint8Array([0x61, 0x00, 0x62, 0x00]),
			Buffer.concat([Buffer.from("한", "utf8"), new Uint8Array([0x00]), Buffer.from("글", "utf8")]),
			new Uint8Array([0x00]),
			new Uint8Array([0x00, 0x00, 0x00, 0x00]),
		];
		for (const bytes of cases) assertSameOutcome(bytes);
	});
});

/** Deterministic PRNG (mulberry32) so property failures reproduce exactly. */
function mulberry32(seed: number): () => number {
	let state = seed;
	return () => {
		state = (state + 0x6d2b79f5) | 0;
		let t = Math.imul(state ^ (state >>> 15), 1 | state);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

type ByteGenerator = (random: () => number, length: number) => Uint8Array;

function randomBytes(random: () => number, length: number, pick: (random: () => number) => number): Uint8Array {
	const bytes = new Uint8Array(length);
	for (let i = 0; i < length; i += 1) bytes[i] = pick(random);
	return bytes;
}

/** Every byte in 0x01-0x7F, ESC included, so the fast path boundary is exercised. */
function asciiBytes(random: () => number, length: number): Uint8Array {
	return randomBytes(random, length, () => 1 + Math.floor(random() * 127));
}

/** Mostly ASCII, with a sprinkle of nulls and high bytes crossing into legacy-path territory. */
function mixedBytes(random: () => number, length: number): Uint8Array {
	return randomBytes(random, length, () => {
		const roll = random();
		if (roll < 0.8) return 1 + Math.floor(random() * 127);
		if (roll < 0.9) return 0;
		return 128 + Math.floor(random() * 128);
	});
}

/** Uniform 0x00-0xFF, the adversarial corner: arbitrary binary data. */
function uniformBytes(random: () => number, length: number): Uint8Array {
	return randomBytes(random, length, () => Math.floor(random() * 256));
}

/**
 * Biased toward valid UTF-8 text (Korean, Latin-1 accents, CJK) with occasional invalid bytes,
 * approximating real-world markdown corpora in several scripts.
 */
function utf8ishBytes(random: () => number, length: number): Uint8Array {
	const fragments = ["한", "글", "문", "서", "é", "ü", "ñ", "日", "中", "Ж", "α", " ", "\n", "\t", "a", "9", "."];
	const parts: Uint8Array[] = [];
	let produced = 0;
	while (produced < length) {
		if (random() < 0.12) {
			// A raw invalid byte: a lone high byte or a forbidden lead (0xC0-0xC1, 0xF5-0xFF).
			const invalid = new Uint8Array([
				random() < 0.5 ? 0x80 + Math.floor(random() * 64) : 0xc0 + Math.floor(random() * 2),
			]);
			parts.push(invalid);
			produced += 1;
		} else {
			const fragment = Buffer.from(fragments[Math.floor(random() * fragments.length)] ?? "a", "utf8");
			parts.push(fragment);
			produced += fragment.length;
		}
	}
	return Buffer.concat(parts).subarray(0, length);
}

function exerciseProperty(seed: number, count: number, generate: ByteGenerator, label: string): void {
	const random = mulberry32(seed);
	let fastPathHits = 0;
	for (let i = 0; i < count; i += 1) {
		const length = Math.floor(random() * 256);
		const bytes = generate(random, length);
		const fast = outcomeOf(decodeText, bytes);
		const reference = outcomeOf(referenceDecodeText, bytes);
		expect(fast, `${label} sample ${i}`).toEqual(reference);
		if (fast.kind === "ok" && isFastPathEligible(bytes)) {
			fastPathHits += 1;
		}
	}
	// Every sample of this generator is fast-path eligible, so a batch that never takes the fast
	// path means the predicate changed and the boundary moved out from under this test.
	if (label === "ascii") {
		expect(fastPathHits, `${label} must exercise the fast path`).toBeGreaterThan(0);
	}
}

function isFastPathEligible(bytes: Uint8Array): boolean {
	if (bytes.length === 0) return true;
	if (bytes.length < 4) return false;
	return bytes.every((byte) => byte >= 0x01 && byte <= 0x7f && byte !== 0x1b);
}

describe("decodeText fast path seeded property equivalence", () => {
	it("matches the pre-change implementation on pure-ASCII buffers", () => {
		exerciseProperty(0xa11ce, 400, asciiBytes, "ascii");
	});

	it("matches on mixed ASCII/null/high-byte buffers", () => {
		exerciseProperty(0xbeef, 400, mixedBytes, "mixed");
	});

	it("matches on arbitrary 0x00-0xFF buffers", () => {
		exerciseProperty(0xdead, 400, uniformBytes, "uniform");
	});

	it("matches on UTF-8-biased text buffers", () => {
		exerciseProperty(0xf00d, 400, utf8ishBytes, "utf8ish");
	});
});

describe("decodeText fast path observability", () => {
	it("skips detect() for pure ASCII and still consults it otherwise", () => {
		detectImpl.current = () => {
			throw new Error("detect must not be called");
		};
		try {
			// Pure ASCII without ESC: the fast path answers without chardet.
			expect(decodeText(Buffer.from("pure ascii body\nline two\n"))).toBe("pure ascii body\nline two\n");
			expect(decodeText(new Uint8Array(0))).toBe("");
			// ESC routes to the legacy path, which calls detect() and must hit the throwing mock.
			expect(() => decodeText(Buffer.from("\x1b$B"))).toThrow("detect must not be called");
			// Any non-ASCII byte routes to the legacy path as well.
			expect(() => decodeText(Buffer.from("한글", "utf8"))).toThrow("detect must not be called");
			expect(() => decodeText(new Uint8Array([0x61, 0x00]))).toThrow("detect must not be called");
		} finally {
			detectImpl.current = null;
		}
	});
});
