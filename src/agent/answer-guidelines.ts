/**
 * The single file-path exception to the answer "no file paths" rule
 * (issue #1790). Every prompt and tool description that shapes `answer` uses
 * this exact wording so a caller that can render images receives a markdown
 * image embed whose path is a real result source, while the no-path rule
 * otherwise still stands.
 */
export const ANSWER_IMAGE_EMBED_RULE =
	"**Images**: when a retrieved result is itself an image file (png, jpg, jpeg, gif, webp, svg, bmp, avif, heic, tiff) and showing it answers the question, embed it in `answer` as a markdown image on its own line: `![short description](<absolute source path>)`, followed by its citation, e.g. `![Q3 revenue chart](</data/reports/q3 chart.png>) [2]`. Use only the real source path of a result in `results`/`mapping`. Never invent a path, use relative paths, or embed remote URLs. Apart from these image embeds, the no-file-paths rule stands.";

/**
 * Delta (verification) answers must not re-embed an image the first answer
 * already showed.
 */
export const ANSWER_IMAGE_DELTA_RULE = "An image already embedded in the first answer must not be embedded again.";

/**
 * One numbering space for citations (issue #1788): every `[n]` in `answer`
 * must be a `results[].number` emitted in the same call. Retrieval candidate
 * numbers and first-answer unit numbers are never citation numbers.
 */
export const ANSWER_CITATION_RULE =
	"**Citation numbers**: every bracketed citation [n] in `answer` must be the `number` of a result you emit in `results` of the same call. Numbers shown in retrieval candidates or in an earlier first answer are NOT citation numbers — renumber against your own `results`, and add a result for any evidence you cite.";
