import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { loadSimplexPeerRegistry, rankSimplexPeerTargets } from "../p2p/simplex-server.ts";

export const RECOMMEND_PEER_TARGETS_TOOL_NAME = "recommend_peer_targets";

const recommendPeerTargetsSchema = Type.Object({
	query: Type.String({ description: "Question or topic to match against local peer personas." }),
});

export interface PeerTargetRecommendation {
	readonly alias: string;
	readonly matchedTerms: readonly string[];
	readonly displayName?: string;
	readonly description?: string;
}

export interface RecommendPeerTargetsDetails {
	readonly method: "recommend_peer_targets";
	readonly resultCount: number;
	readonly matches: readonly PeerTargetRecommendation[];
}

export function createRecommendPeerTargetsTool(
	workspacePath: string,
): AgentTool<typeof recommendPeerTargetsSchema, RecommendPeerTargetsDetails> {
	return {
		name: RECOMMEND_PEER_TARGETS_TOOL_NAME,
		label: "Recommend Peer Targets",
		description:
			"Recommend local peer personas to ask about a topic using explainable keyword matches. Read-only; never contacts peers.",
		parameters: recommendPeerTargetsSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<RecommendPeerTargetsDetails>> {
			const registry = loadSimplexPeerRegistry(workspacePath);
			const matches = rankSimplexPeerTargets(params.query, registry).map((match) => {
				const peer = registry[match.alias];
				return {
					alias: match.alias,
					matchedTerms: match.matchedTerms,
					...(peer?.displayName !== undefined ? { displayName: peer.displayName } : {}),
					...(peer?.description !== undefined ? { description: peer.description } : {}),
				};
			});
			const details: RecommendPeerTargetsDetails = {
				method: "recommend_peer_targets",
				resultCount: matches.length,
				matches,
			};
			return {
				content: [{ type: "text", text: JSON.stringify(details) }],
				details,
			};
		},
	};
}
