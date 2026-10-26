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
 * Answer readability rules. Every surface that tells the model how to write the
 * answer (system prompt, fast/complete/delta/search phase prompts, report
 * schema) carries the same wording, so the answer's shape does not depend on
 * which phase produced it. Each rule yields to an explicit user request for a
 * different format.
 */

/**
 * Conclusion first. Only complete answers get a lead line: a delta answer holds
 * only corrections and new findings, and a lead conclusion would restate the
 * first answer. The lead is outside the 5-bullet cap so the cap stays intact.
 */
export const ANSWER_LEAD_RULE =
	"**Lead with the conclusion**: unless the user explicitly requests a different format, start the answer with one line that states the conclusion in **bold**, followed by the supporting bullet points. The lead line is not counted in the 5-bullet cap. When the whole answer is a single fact, the lead line alone is enough and no bullets are needed.";

/** One claim per bullet; flat list. */
export const ANSWER_BULLET_RULE =
	"**One claim per bullet**: each bullet makes one claim in one or two sentences. Never nest bullets; split a bullet that carries two claims into two bullets.";

/** Bold the values a reader scans for; tabulate real comparisons. */
export const ANSWER_EMPHASIS_RULE =
	"**Emphasis**: put the values a reader scans for (dates, amounts, names, statuses, deadlines) in **bold**, and nothing else. When comparing three or more items across the same attributes, use a markdown table instead of bullets.";

/** Answer language follows the question, not the sources. */
export const ANSWER_LANGUAGE_RULE =
	"**Answer language**: write the answer in the language of the user's question (a Korean question gets a Korean answer) even when the sources are in another language; if the user asks for a specific language, use that one. Keep proper nouns, identifiers, code, and terms quoted from sources in their original form.";

/** Uncertainty goes last so it never buries the conclusion. */
export const ANSWER_UNCERTAINTY_RULE =
	"**Uncertainty placement**: when the answer is incomplete or uncertain, state that in one short sentence on the last line of the answer, never before or ahead of the conclusion. Do not hedge the conclusion itself.";

/**
 * Every sentence answers the question. Models that find nothing on point tend to
 * prove the search was thorough by listing what they found instead and why it
 * does not match; the caller never asked for that, so it is left out for every
 * phase, delta included.
 */
export const ANSWER_RELEVANCE_RULE =
	'**Answer only what was asked**: every sentence must answer the user\'s question. Never list retrieved items, topics, documents, or figures that do not answer it, never describe how or where you searched, and never explain why something was left out or how it differs from what was asked (no "I also found X, but it is not what you asked", no near-miss lists, no "nothing about Y, only Z", no inventory of the sources that lacked it). Unrelated or merely adjacent material is omitted entirely unless the user asked for related items or alternatives.';

/**
 * Fast phase, no supporting evidence: park the question in one line. The
 * verification phase that follows keeps searching, so the line is true.
 */
export const ANSWER_UNCONFIRMED_FAST_RULE =
	'**When the evidence does not answer the question yet**: do not build an answer out of what was found instead, and do not argue that the answer is absent by naming what the sources did contain (no "only X exists" contrast). The whole answer is then one short line saying it is not confirmed yet and that you are searching further (the verification phase that follows does the searching), with no bullets and no citations. A different subject, place, year, or party is unrelated evidence, not part of the answer. If a part the user actually asked for is answered, answer that part normally and say on the last line what is still being searched.';

/**
 * Phases that can end the run: search before giving up, and when the search
 * still finds nothing the answer is one line, not an inventory of the corpus.
 */
export const ANSWER_UNCONFIRMED_FINAL_RULE =
	'**When the answer cannot be found**: keep searching first (different terms, synonyms, other retrieval methods or datasources) instead of answering from unrelated results. If it still cannot be confirmed, the whole answer is one short line saying so, with no bullets, no citations, no mention of what was found instead (no "only X exists" contrast), and no account of where you looked or what the documents do contain. A different subject, place, year, or party is unrelated evidence, not part of the answer. If a part the user actually asked for is answered, answer that part normally and say on the last line what is still unconfirmed.';

/**
 * Inline evidence citations. The model writes the id the retrieval tools print
 * next to each result; the harness resolves it to the recorded source and
 * renumbers the markers `[1]`, `[2]`, ... (a source keeps its number for the whole
 * conversation), so the caller only ever sees numbered citations backed by recorded evidence.
 */
export const ANSWER_CITATION_RULE =
	'**Citations**: cite supporting evidence inline with the evidence id shown next to each retrieved result, in square brackets (e.g. [e3], or [e3][e7] for two). Copy ids exactly; never invent one. For a local file you opened yourself with `bash`, cite `[file:<absolute path>]`. Cite the evidence that backs a sentence at the end of that sentence. Do not cite numbers, paths, or anything else. A retrieved result may carry a measured location note such as `(lines 7-8, section "Article 5 (Member Obligations)")` next to its id: when you state a line number, line range, or article/section of a passage, copy it from that note, and when no note is printed give none. Never count or infer line numbers yourself, and never state how many items (paragraphs, articles, results) there are unless you counted them in the quoted text.';

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
	'**Deliverables and artifacts**: if the user asks you to draft, write, generate, or prepare a deliverable or artifact (a reply email, letter, legal opinion, summary document, standard-format case citation block, checklist, report, or workflow document — including requests phrased as "how should I reply..." to a specific message), produce the finished, ready-to-use artifact as a file (using `write`, `edit`, or `bash` in local sessions) in a file format, extension, and structure aligned with the context and domain (e.g. `.eml` or `.md`/`.txt` with headers for an email reply; `.md`, `.txt`, `.docx`, `.hwp`, or `.pdf` matching workspace conventions for a legal opinion, insertable case-citation block, or summary document; `.csv`, `.xlsx`, or `.json` for structured tabular data), containing the complete unabridged artifact (including required headers, precedent passages, pinpoint paragraphs, and standard-format citations) inside the file instead of merely describing how to produce it. Do NOT dump the full artifact text into your reply: unless the user explicitly requests otherwise, keep your reply concise within the at-most-5-bullet direct-answer format, clearly stating the created file\'s exact path/location and briefly summarizing what was generated with inline citations to the underlying source evidence (`[eN]` or `[file:<source path>]`); never cite the newly created deliverable file itself as `[file:...]`.';

/**
 * Verification-phase deliverable rule: create the context-aligned deliverable
 * file (in local sessions) and report its location and concise summary if the
 * fast-phase draft could not write the file yet or only gave advice about it.
 */
export const ANSWER_DELIVERABLE_DELTA_RULE =
	"If a deliverable or artifact was requested and the first answer could not write the file yet (the fast phase runs without tool calls) or only gave advice about it, create the complete context-aligned deliverable file now during verification and report the created file's location and a brief summary concisely.";
