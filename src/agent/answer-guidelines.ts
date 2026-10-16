/**
 * The single file-path exception to the answer "no file paths" rule
 * (issue #1790). Every prompt that shapes the answer uses this exact wording so
 * a caller that can render images receives a markdown image embed whose path is
 * a real evidence source, while the no-path rule otherwise still stands.
 */
export const ANSWER_IMAGE_EMBED_RULE =
	"**Images**: when a retrieved result is itself an image file (png, jpg, jpeg, gif, webp, svg, bmp, avif, heic, tiff) and showing it answers the question, embed it in the answer as a markdown image on its own line: `![short description](<absolute source path>)`, followed by its citation, e.g. `![Q3 revenue chart](</data/reports/q3 chart.png>) [e2]`. Use only the real source path of retrieved evidence. Never invent a path, use relative paths, or embed remote URLs. Apart from these image embeds, the no-file-paths rule stands.";

/**
 * Delta (verification) answers must not re-embed an image the first answer
 * already showed.
 */
export const ANSWER_IMAGE_DELTA_RULE = "An image already embedded in the first answer must not be embedded again.";

/**
 * Inline evidence citations. The model writes the id the retrieval tools print
 * next to each result; the harness resolves it to the recorded source and
 * renumbers the markers `[1]`, `[2]`, ... in order of first appearance, so the
 * caller only ever sees numbered citations backed by recorded evidence.
 */
export const ANSWER_CITATION_RULE =
	"**Citations**: cite supporting evidence inline with the evidence id shown next to each retrieved result, in square brackets (e.g. [e3], or [e3][e7] for two). Copy ids exactly; never invent one. For a local file you opened yourself with `bash`, cite `[file:<absolute path>]`. Cite the evidence that backs a sentence at the end of that sentence. Do not cite numbers, paths, or anything else.";

/**
 * Deliverable/artifact rule (issue #1832). The 5-bullet cap and direct-answer
 * format remain the default unless the user explicitly asks otherwise; when the
 * caller requests a deliverable (a reply email, letter, legal opinion, summary
 * document, insertable case citation block, checklist, report, or workflow
 * document), the agent produces the complete finished artifact as a file in a
 * format aligned with the context and domain, and keeps the answer itself
 * concise by stating the created file's location and briefly summarizing what
 * was generated (citing only the underlying source evidence).
 */
export const ANSWER_DELIVERABLE_RULE =
	'**Deliverables and artifacts**: if the user asks you to draft, write, generate, or prepare a deliverable or artifact (a reply email, letter, legal opinion, summary document, standard-format case citation block, checklist, report, or workflow document — including requests phrased as "how should I reply..." to a specific message), produce the finished, ready-to-use artifact as a file (using `write`, `edit`, or `bash` in local sessions, without overwriting existing source files) in a file format, extension, and structure aligned with the context and domain (e.g. `.eml` or `.md`/`.txt` with headers for an email reply; `.md`, `.txt`, `.docx`, `.hwp`, or `.pdf` matching workspace conventions for a legal opinion, insertable case-citation block, or summary document; `.csv`, `.xlsx`, or `.json` for structured tabular data), containing the complete unabridged artifact (including required headers, precedent passages, pinpoint paragraphs, and standard-format citations) inside the file instead of merely describing how to produce it. Do NOT dump the full artifact text into your reply: unless the user explicitly requests otherwise, keep your reply concise within the at-most-5-bullet direct-answer format, clearly stating the created file\'s exact path/location and briefly summarizing what was generated with inline citations to the underlying source evidence (`[eN]` or `[file:<source path>]`); never cite the newly created deliverable file itself as `[file:...]`.';

/**
 * Verification-phase deliverable rule: create the context-aligned deliverable
 * file (in local sessions) and report its location and concise summary if the
 * fast-phase draft could not write the file yet or only gave advice about it.
 */
export const ANSWER_DELIVERABLE_DELTA_RULE =
	"If a deliverable or artifact was requested and the first answer could not write the file yet (the fast phase runs without tool calls) or only gave advice about it, create the complete context-aligned deliverable file now during verification and report the created file's location and a brief summary concisely.";
