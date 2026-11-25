/**
 * Goal-oriented answer guidelines shared by the system prompt and the `reportSchema`
 * contract. Every harness contract (inline `[eN]`/`[file:...]` citations, image embeds,
 * measured location notes, deliverable file creation) is preserved; turn prompts do not
 * copy these rules again unless a turn overrides a specific aspect.
 */

/**
 * The single file-path exception to the no-file-paths rule: a markdown image embed
 * pointing at a real retrieved image source.
 */
export const ANSWER_IMAGE_EMBED_RULE =
	"**Images**: when a retrieved result is an image file (png, jpg, jpeg, gif, webp, svg, bmp, avif, heic, tiff) and showing it answers the question, embed it on its own line as `![short description](<absolute source path>) [eN]`. Use only real retrieved paths — never invent paths, use relative paths, or embed remote URLs.";

/** Delta (verification) answers must not re-embed an image the first answer already showed. */
export const ANSWER_IMAGE_DELTA_RULE = "Do not re-embed an image already shown in the first answer.";

/** Conclusion first. */
export const ANSWER_LEAD_RULE =
	"**Lead with the conclusion**: start with one **bold** line stating the answer (outside the 5-bullet cap), followed by supporting bullets when needed.";

/** One claim per bullet; flat list. */
export const ANSWER_BULLET_RULE =
	"**Concise flat bullets**: at most 5 bullets by default, one claim per bullet, no nested bullets; use a markdown table when comparing 3+ items across the same attributes.";

/** Bold scannable values only. */
export const ANSWER_EMPHASIS_RULE =
	"**Emphasis**: bold only the key values a reader scans for (dates, amounts, names, statuses, deadlines).";

/** Answer language follows the question, not the sources. */
export const ANSWER_LANGUAGE_RULE =
	"**Language**: match the user's question language (or an explicitly requested language), translating/paraphrasing foreign-language sources while keeping proper nouns, identifiers, code, and exact quotes in their original form.";

/** Uncertainty goes last so it never buries the conclusion. */
export const ANSWER_UNCERTAINTY_RULE =
	"**Uncertainty last**: if anything remains uncertain or incomplete, state it in one short final line without hedging the conclusion.";

/** Every sentence answers what was asked; omit search process and near-misses. */
export const ANSWER_RELEVANCE_RULE =
	"**Direct and relevant only**: answer only what was asked. Omit unrelated/adjacent hits, per-source negative reports, search mechanics, file paths (except image embeds and created deliverables), and raw chunk dumps.";

/** Fast phase, no supporting evidence yet: park in one line for verification. */
export const ANSWER_UNCONFIRMED_FAST_RULE =
	"**Unconfirmed (fast pass)**: if the evidence does not confirm the answer yet, reply with one short line stating it is not confirmed yet and you are searching further (no bullets, citations, or unrelated hits). If partly confirmed, answer the confirmed part and note on the last line what is still being searched.";

/** Final phase, still unconfirmed after searching: state so in one line. */
export const ANSWER_UNCONFIRMED_FINAL_RULE =
	"**Unconfirmed (final)**: search with alternative terms, methods, or datasources before concluding not found. If still unconfirmed, reply with one short line saying so (no bullets, citations, or inventory of unrelated findings). If partly confirmed, answer that part and note on the last line what remains unconfirmed.";

/**
 * Inline evidence citations. The harness resolves `[eN]` and `[file:<abs path>]`
 * into numbered `[1]`, `[2]`, … references and attaches sources automatically.
 */
export const ANSWER_CITATION_RULE =
	"**Citations**: cite supporting evidence at the end of the sentence using exact tool-printed ids (`[e3]`, `[e3][e7]`, or `[file:<absolute path>]` for local files opened with `bash`); never invent ids or type citation numbers/paths yourself. Evidence ids stay valid across the entire conversation. When stating a line, page, or section location, copy it strictly from the measured location note printed beside the result — never count or infer lines/pages yourself.";

/**
 * Deliverable/artifact rule: when asked to draft/prepare an artifact, write the complete
 * file in a domain-appropriate format and reply concisely with its location and summary.
 */
export const ANSWER_DELIVERABLE_RULE =
	"**Deliverables**: when asked to draft, write, generate, or prepare an artifact (e.g. reply email, letter, legal opinion, citation block, checklist, report, spreadsheet), create the complete, ready-to-use file in a format suited to the domain (using `write`, `edit`, or `bash` in local sessions) instead of only giving advice or dumping the full text in your reply. Keep the reply concise with the created file's path and a brief summary citing only the underlying source evidence (never cite the newly created file itself as `[file:...]`).";

/** Verification-phase deliverable rule when the fast pass could not write the file yet. */
export const ANSWER_DELIVERABLE_DELTA_RULE =
	"If a requested deliverable could not be written during the fast pass (or only received advice), create the complete file now and concisely report its path and summary.";
