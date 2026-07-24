export const MIRACL_SOURCES = {
	topics: {
		revision: "5be20db9509754dadad47689368639fcec739c00",
		topicsUrl:
			"https://huggingface.co/datasets/miracl/miracl/resolve/5be20db9509754dadad47689368639fcec739c00/miracl-v1.0-ko/topics/topics.miracl-v1.0-ko-dev.tsv",
		qrelsUrl:
			"https://huggingface.co/datasets/miracl/miracl/resolve/5be20db9509754dadad47689368639fcec739c00/miracl-v1.0-ko/qrels/qrels.miracl-v1.0-ko-dev.tsv",
	},
	corpus: {
		revision: "d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
		urls: [0, 1, 2].map(
			(shard) =>
				`https://huggingface.co/datasets/miracl/miracl-corpus/resolve/d921ec7e349ce0d28daf30b2da9da5ee698bef0d/miracl-corpus-v1.0-ko/docs-${shard}.jsonl.gz`,
		),
	},
} as const;

export const MIRACL_SMOKE_PROFILE = {
	seed: 20260723,
	queryCount: 32,
	distractorCount: 10_000,
} as const;

export const MIRACL_NORMALIZATION_VERSION = 1 as const;
