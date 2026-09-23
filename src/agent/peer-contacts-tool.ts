import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { loadSimplexPeerRegistry, type SimplexPeerRecord, saveSimplexPeerRegistry } from "../p2p/simplex-server.ts";

export const LIST_PEER_CONTACTS_TOOL_NAME = "list_peer_contacts";
export const UPDATE_PEER_CONTACT_DESCRIPTION_TOOL_NAME = "update_peer_contact_description";

/** Local background notes stay short enough to reread before every peer query. */
export const MAX_PEER_DESCRIPTION_LENGTH = 2000;

const CONTROL_CHARACTERS = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/u;

const listPeerContactsSchema = Type.Object({});

const updatePeerContactDescriptionSchema = Type.Object({
	alias: Type.String({
		minLength: 1,
		description: "Registry alias of the trusted contact, as returned by list_peer_contacts.",
	}),
	description: Type.String({
		description:
			"Local background note: who this person is and which documents their AutoRAG agent holds. Not sent to them. An empty string clears the note. Maximum 2000 characters.",
	}),
});

export interface PeerContactRecord {
	readonly alias: string;
	readonly contactId: number;
	readonly addedAt: string;
	readonly displayName?: string;
	readonly description?: string;
	readonly role?: string;
	readonly org?: string;
	readonly accessHint?: readonly string[];
	readonly descriptionMissing: boolean;
}

export interface ListPeerContactsDetails {
	readonly method: "list_peer_contacts";
	readonly resultCount: number;
	readonly contacts: readonly PeerContactRecord[];
}

export interface UpdatePeerContactDescriptionDetails {
	readonly method: "update_peer_contact_description";
	readonly ok: boolean;
	readonly alias: string;
	readonly description?: string;
	readonly message: string;
}

function contactRecord(alias: string, peer: SimplexPeerRecord): PeerContactRecord {
	const description = peer.description?.trim();
	return {
		alias,
		contactId: peer.contactId,
		addedAt: peer.addedAt,
		...(peer.displayName !== undefined ? { displayName: peer.displayName } : {}),
		...(description !== undefined && description.length > 0 ? { description } : {}),
		...(peer.role !== undefined ? { role: peer.role } : {}),
		...(peer.org !== undefined ? { org: peer.org } : {}),
		...(peer.accessHint !== undefined ? { accessHint: peer.accessHint } : {}),
		descriptionMissing: description === undefined || description.length === 0,
	};
}

function result<T>(details: T, text = JSON.stringify(details)): AgentToolResult<T> {
	return { content: [{ type: "text", text }], details };
}

/**
 * Read every trusted SimpleX contact, including contacts that have no
 * background description yet. Does not open a network connection.
 */
export function createListPeerContactsTool(
	workspacePath: string,
): AgentTool<typeof listPeerContactsSchema, ListPeerContactsDetails> {
	return {
		name: LIST_PEER_CONTACTS_TOOL_NAME,
		label: "List Peer Contacts",
		description:
			"List every trusted SimpleX contact in the local peer registry: alias, contact id, and the operator's background description of who that person is and what documents they hold. Read-only and local; it does not contact anyone. Use this before choosing who to query. A contact with descriptionMissing true has no background note yet.",
		parameters: listPeerContactsSchema,
		async execute(): Promise<AgentToolResult<ListPeerContactsDetails>> {
			const contacts = Object.entries(loadSimplexPeerRegistry(workspacePath)).map(([alias, peer]) =>
				contactRecord(alias, peer),
			);
			return result({
				method: "list_peer_contacts",
				resultCount: contacts.length,
				contacts,
			});
		},
	};
}

/**
 * Replace or clear one contact's local background description.
 * Trust (contact id) and every other contact stay unchanged.
 */
export function createUpdatePeerContactDescriptionTool(
	workspacePath: string,
): AgentTool<typeof updatePeerContactDescriptionSchema, UpdatePeerContactDescriptionDetails> {
	return {
		name: UPDATE_PEER_CONTACT_DESCRIPTION_TOOL_NAME,
		label: "Update Peer Contact Description",
		description:
			"Write, replace, or clear the local background description for one trusted contact alias. The description records who that person is and which documents their AutoRAG agent holds. It stays on this machine and is never sent to the peer. Pass an empty description to clear the note. Fails, without changing the registry, when the alias is unknown, the note is longer than 2000 characters, or it contains control characters. This does not add a contact or change their contact id.",
		parameters: updatePeerContactDescriptionSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<UpdatePeerContactDescriptionDetails>> {
			const alias = params.alias;
			const registry = loadSimplexPeerRegistry(workspacePath);
			const existing = registry[alias];
			if (existing === undefined) {
				return result({
					method: "update_peer_contact_description",
					ok: false,
					alias,
					message: `Peer not found: ${alias}. Use list_peer_contacts for the aliases that exist.`,
				});
			}
			const description = params.description.trim();
			if (description.length > MAX_PEER_DESCRIPTION_LENGTH) {
				return result({
					method: "update_peer_contact_description",
					ok: false,
					alias,
					message: `Description is ${description.length} characters; the maximum is ${MAX_PEER_DESCRIPTION_LENGTH}. The registry was not changed.`,
				});
			}
			if (CONTROL_CHARACTERS.test(description)) {
				return result({
					method: "update_peer_contact_description",
					ok: false,
					alias,
					message: "Description contains control characters. The registry was not changed.",
				});
			}
			const { description: _previous, ...rest } = existing;
			registry[alias] = description.length === 0 ? rest : { ...existing, description };
			saveSimplexPeerRegistry(workspacePath, registry);
			return result({
				method: "update_peer_contact_description",
				ok: true,
				alias,
				...(description.length > 0 ? { description } : {}),
				message:
					description.length === 0
						? `Cleared the local background description for ${alias}.`
						: `Saved the local background description for ${alias}.`,
			});
		},
	};
}
