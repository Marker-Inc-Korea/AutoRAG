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
 * Deliverable/artifact exception to the 5-bullet cap and question-answering
 * shape (issue #1832). When the caller asks for a finished deliverable (a reply
 * email, letter, legal opinion, summary document, insertable case citation,
 * checklist, or workflow summary), every prompt that shapes the answer instructs
 * the model to produce the finished artifact itself rather than advice about how
 * to write it.
 */
export const ANSWER_DELIVERABLE_RULE =
	'**Deliverables and artifacts**: if the user asks you to draft, write, generate, or prepare something (a reply, letter, email, opinion, summary document, citation block, checklist, or workflow summary) — including requests phrased as "how should I reply..." to a specific message — put the finished artifact itself in the answer — ready to use — instead of describing how to produce it. The 5-bullet cap and the "direct answer only" wording do not apply to a requested artifact; keep paths and retrieval mechanics out, but do not compress away the artifact or omit required precedent passages, paragraphs, or standard-format citations.';

/**
 * Delta (verification) answers must still deliver the full artifact if the
 * first answer omitted it or only gave advice about it.
 */
export const ANSWER_DELIVERABLE_DELTA_RULE =
	"If the first answer omitted a requested artifact or only described how to write it, output the complete ready-to-use artifact now.";
