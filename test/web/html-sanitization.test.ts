import { describe, expect, it } from "vitest";
import { decodeHtmlEntities } from "../../src/web/entities.ts";
import { cleanFeedText } from "../../src/web/fetch/feeds.ts";
import { htmlToBasicMarkdown } from "../../src/web/fetch/html-renderer.ts";
import TurndownService from "../../src/web/fetch/turndown/service.ts";

describe("decodeHtmlEntities (single-pass, CodeQL js/double-escaping)", () => {
	it("decodes named and numeric entities", () => {
		expect(decodeHtmlEntities("&lt;b&gt; &amp; &quot;q&quot; &#39; &nbsp;x")).toBe('<b> & "q" \'  x');
	});

	it("never double-unescapes: an entity produced by a decode is NOT re-decoded", () => {
		// `&amp;lt;` encodes the literal text `&lt;` — it must not collapse to `<`.
		expect(decodeHtmlEntities("&amp;lt;")).toBe("&lt;");
		// `&#38;#60;` encodes `&#60;` — it must not collapse to `<`.
		expect(decodeHtmlEntities("&#38;#60;")).toBe("&#60;");
	});

	it("leaves unknown entities and bare ampersands untouched", () => {
		expect(decodeHtmlEntities("a & b &notanentity; &fake;")).toBe("a & b &notanentity; &fake;");
	});
});

describe("cleanFeedText (CodeQL js/incomplete-multi-character-sanitization)", () => {
	it("unwraps a full CDATA wrapper and strips tags", () => {
		expect(cleanFeedText("<![CDATA[<p>Hello <b>world</b></p>]]>")).toBe("Hello world");
	});

	it("does not eat a stray CDATA terminator inside content", () => {
		expect(cleanFeedText("a ]]> b")).toBe("a ]]> b");
	});
});

describe("htmlToBasicMarkdown script/style removal (CodeQL js/bad-tag-filter)", () => {
	it("drops script and style content", () => {
		const md = htmlToBasicMarkdown("<p>Hello</p><script>alert(1)</script><style>.x{color:red}</style>");
		expect(md).toBe("Hello");
	});

	it("a nested/broken script tag cannot leak script markup through", () => {
		// Parser-level removal: broken nesting can leave inert TEXT behind
		// (the output feeds an LLM, never an HTML renderer), but no script
		// markup may survive as markdown.
		const md = htmlToBasicMarkdown("<scr<script>ipt>alert(1)</scr</script>ipt><p>safe</p>");
		expect(md).not.toMatch(/<\/?script/i);
		expect(md).toContain("safe");
	});
});

describe("turndown link/image destination escaping (CodeQL js/incomplete-sanitization)", () => {
	it("escapes backslashes before escaping markdown-sensitive characters", () => {
		const turndown = new TurndownService();
		const md = turndown.turndown('<a href="https://example.com/a\\b(c)">t</a>');
		expect(md).toBe("[t](https://example.com/a\\\\b\\(c\\))");
	});

	it("escapes backslashes in link titles before quoting", () => {
		const turndown = new TurndownService();
		const md = turndown.turndown(`<a href="https://example.com" title='a\\"b'>t</a>`);
		expect(md).toBe('[t](https://example.com "a\\\\\\"b")');
	});
});
