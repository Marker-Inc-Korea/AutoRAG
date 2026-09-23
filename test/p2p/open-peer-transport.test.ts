import { describe, expect, it } from "vitest";
import { openPeerQueryTransport, type SimplexTransport } from "../../src/p2p/simplex-transport.ts";

function fakeTransport(label: string): SimplexTransport {
	return {
		dbPrefix: label,
		displayName: label,
		async getUserId() {
			return 1;
		},
		async getOrCreateAddress() {
			return label;
		},
		async createInvitation() {
			return label;
		},
		async connect() {},
		async listContacts() {
			return [];
		},
		async sendMessage() {},
		onMessage() {
			return () => {};
		},
		async close() {},
	};
}

describe("openPeerQueryTransport", () => {
	it("attaches to a listening simplex-chat port and does not spawn", async () => {
		const calls: string[] = [];
		const attached = fakeTransport("attached");
		const transport = await openPeerQueryTransport({
			dbPrefix: "/tmp/simplex",
			displayName: "autorag",
			port: 5225,
			portOpen: async () => {
				calls.push("probe");
				return true;
			},
			start: async () => {
				calls.push("start");
				return fakeTransport("started");
			},
			attach: async () => {
				calls.push("attach");
				return attached;
			},
		});
		expect(calls).toEqual(["probe", "attach"]);
		expect(transport).toBe(attached);
	});

	it("starts simplex-chat only when the port is closed", async () => {
		const calls: string[] = [];
		const started = fakeTransport("started");
		const transport = await openPeerQueryTransport({
			dbPrefix: "/tmp/simplex",
			displayName: "autorag",
			port: 5225,
			portOpen: async () => {
				calls.push("probe");
				return false;
			},
			start: async () => {
				calls.push("start");
				return started;
			},
			attach: async () => {
				calls.push("attach");
				return fakeTransport("attached");
			},
		});
		expect(calls).toEqual(["probe", "start"]);
		expect(transport).toBe(started);
	});
});
