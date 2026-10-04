import {
	type Answer,
	choice as choiceQuestion,
	createBuiltinJevModels,
	type Entry,
	type JevModel,
	type JevResult,
	type MutableJevModels,
	noul as noulQuestion,
	type Question,
	type Questions,
	score as scoreQuestion,
} from "@geminixiang/jev";
import {
	DEFAULT_JEV_MODEL,
	JEV_BACKENDS,
	type JevAnswerSummary,
	type JevBackend,
	type JevEvaluation,
	type JevEvaluator,
	JevInputError,
	type JevQuestion,
	type JevToolOptions,
	MAX_JEV_QUESTIONS,
} from "./types.ts";

function buildQuestion(name: string, question: JevQuestion): Question {
	switch (question.type) {
		case "noul": {
			// OpenRouter rejects a partial or empty criteria object, so only forward
			// the outcome descriptions when both are present.
			const criteria =
				question.true !== undefined && question.false !== undefined
					? { true: question.true, false: question.false }
					: undefined;
			return noulQuestion(question.instructions, criteria);
		}
		case "choice": {
			const criteria = Object.entries(question.criteria);
			if (criteria.length === 0) {
				throw new JevInputError(`Jev question "${name}" is a choice but has no criteria options`);
			}
			return choiceQuestion(question.instructions, Object.fromEntries(criteria));
		}
		case "score": {
			const [first, second, ...rest] = question.criteria;
			if (first === undefined || second === undefined) {
				throw new JevInputError(`Jev question "${name}" is a score but has fewer than two criteria levels`);
			}
			return scoreQuestion(question.instructions, [first, second, ...rest]);
		}
		default:
			// The tool parser rejects unknown `type` values before this point.
			throw new JevInputError(`Jev question "${name}" has an unsupported type`);
	}
}

function buildQuestions(questions: Readonly<Record<string, JevQuestion>>): Questions {
	const names = Object.keys(questions);
	if (names.length === 0) {
		throw new JevInputError("At least one Jev question is required");
	}
	if (names.length > MAX_JEV_QUESTIONS) {
		throw new JevInputError(`At most ${MAX_JEV_QUESTIONS} Jev questions are allowed per call`);
	}
	const built: Record<string, Question> = {};
	for (const name of names) {
		built[name] = buildQuestion(name, questions[name]);
	}
	return built;
}

function summarizeAnswer(answer: Answer): JevAnswerSummary {
	switch (answer.type) {
		case "noul":
			return { type: "noul", noul: answer.noul };
		case "choice":
			return {
				type: "choice",
				choice: answer.choice,
				...(answer.confidence !== undefined ? { confidence: answer.confidence } : {}),
				probabilities: { ...answer.probabilities },
			};
		case "score":
			return {
				type: "score",
				score: answer.score,
				...(answer.confidence !== undefined ? { confidence: answer.confidence } : {}),
				probabilities: { ...answer.probabilities },
			};
	}
}

function summarizeEvaluation(result: JevResult): JevEvaluation {
	const answers: Record<string, JevAnswerSummary> = {};
	for (const [name, answer] of Object.entries(result.answers)) {
		answers[name] = summarizeAnswer(answer);
	}
	return {
		provider: String(result.provider),
		model: result.model,
		answers,
		usage: {
			inputTokens: result.usage.input,
			outputTokens: result.usage.output,
			totalTokens: result.usage.totalTokens,
			costUsd: result.usage.cost.total,
		},
	};
}

async function resolveModel(
	models: MutableJevModels,
	backend: JevBackend | undefined,
	modelId: string,
): Promise<JevModel> {
	if (backend !== undefined) {
		const direct = models.getModel(backend, modelId);
		if (direct !== undefined) return direct;
		const available = await models.getAvailable(backend);
		const fallback = available[0];
		if (fallback !== undefined) return fallback;
		throw new JevInputError(
			`Jev backend "${backend}" exposes no model "${modelId}" and no authenticated models; check its API key`,
		);
	}
	for (const candidate of JEV_BACKENDS) {
		const available = await models.getAvailable(candidate);
		const match = available.find((model) => model.id === modelId) ?? available[0];
		if (match !== undefined) return match;
	}
	throw new JevInputError(
		"No authenticated Jev backend. Set TYPESAFE_API_KEY, OPENROUTER_API_KEY, AI_GATEWAY_API_KEY/VERCEL_API_KEY, or CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID.",
	);
}

/**
 * SDK-backed {@link JevEvaluator}. The model catalog is created lazily on the
 * first call so an enabled tool never resolves credentials at construction time.
 */
export function createJevEvaluator(options: JevToolOptions = {}): JevEvaluator {
	const modelId = options.model ?? DEFAULT_JEV_MODEL;
	let models: MutableJevModels | undefined;
	return {
		async evaluate(input) {
			models ??= createBuiltinJevModels({
				...(options.timeoutMs !== undefined ? { timeoutMs: options.timeoutMs } : {}),
				...(options.maxRetries !== undefined ? { maxRetries: options.maxRetries } : {}),
			});
			const model = await resolveModel(models, options.backend, modelId);
			const apiKey = options.apiKeyEnv !== undefined ? process.env[options.apiKeyEnv] : undefined;
			const result = await models.evaluate(
				model,
				{ state: input.state as Entry, questions: buildQuestions(input.questions) },
				{
					...(apiKey !== undefined && apiKey.length > 0 ? { apiKey } : {}),
					...(options.timeoutMs !== undefined ? { timeoutMs: options.timeoutMs } : {}),
					...(options.maxRetries !== undefined ? { maxRetries: options.maxRetries } : {}),
					...(input.signal !== undefined ? { signal: input.signal } : {}),
				},
			);
			return summarizeEvaluation(result);
		},
	};
}
