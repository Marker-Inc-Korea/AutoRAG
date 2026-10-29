import { compareWithDupey, DupeyCliError, type DupeyCliOptions, type DupeyCompareResult } from "../dupey/index.ts";
import { convertWithFilee, type FileeCliOptions, type FileeConvertRequest, type FileeConvertResult } from "./cli.ts";

/** What `dupey scan` would say about a converted copy and its source. */
export type ConvertedCopyRelation = DupeyCompareResult | { readonly relation: "unavailable"; readonly reason: string };

export interface ConvertedCopyLink {
	readonly source: string;
	readonly output: string;
	readonly dupey: ConvertedCopyRelation;
}

export interface ConvertAndRelateResult {
	readonly result: FileeConvertResult;
	/** One entry per output file written by a successful conversion. Empty when dupey is disabled. */
	readonly links: readonly ConvertedCopyLink[];
}

export interface ConvertAndRelateOptions {
	readonly filee?: FileeCliOptions;
	/** `false` skips the duplicate-family check. */
	readonly dupey?: DupeyCliOptions | false;
}

/**
 * Convert, then ask dupey whether each output is a duplicate-family member of
 * its source. A converted copy has the same text in another format, which is
 * exactly what dupey's families describe; the comparison uses dupey's own
 * thresholds so it agrees with a later `dupey scan` of the directory.
 * A dupey that cannot run never fails the conversion.
 */
export async function convertAndRelate(
	request: FileeConvertRequest,
	options: ConvertAndRelateOptions = {},
): Promise<ConvertAndRelateResult> {
	const result = await convertWithFilee(request, options.filee);
	if (options.dupey === false) return { result, links: [] };
	const dupeyOptions = options.dupey ?? {};
	const pairs = result.files
		.filter((file) => file.state === "done")
		.flatMap((file) => file.outputs.map((output) => ({ source: file.source, output })));
	const links = await Promise.all(
		pairs.map(async ({ source, output }): Promise<ConvertedCopyLink> => {
			try {
				return { source, output, dupey: await compareWithDupey(source, output, dupeyOptions) };
			} catch (error) {
				if (!(error instanceof DupeyCliError)) throw error;
				return { source, output, dupey: { relation: "unavailable", reason: error.message } };
			}
		}),
	);
	return { result, links };
}
