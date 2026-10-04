import { describe, expect, it, vi } from "vitest";
import {
	AUTORAG_LATEST_VERSION_URL,
	AUTORAG_PACKAGE_NAME,
	AUTORAG_UPDATE_COMMAND,
	type AutoRAGUpdateResult,
	checkAutoRAGUpdate,
	comparePackageVersions,
	renderAutoRAGUpdateNotice,
} from "../../src/cli/update-check.ts";

function jsonResponse(version: string): Response {
	return new Response(JSON.stringify({ version }), {
		status: 200,
		headers: { "content-type": "application/json" },
	});
}

describe("comparePackageVersions", () => {
	it("orders by numeric core and treats releases as newer than prereleases", () => {
		expect(comparePackageVersions("1.2.3", "1.2.2")).toBeGreaterThan(0);
		expect(comparePackageVersions("2.0.0", "1.9.9")).toBeGreaterThan(0);
		expect(comparePackageVersions("1.2.3", "1.2.3")).toBe(0);
		expect(comparePackageVersions("1.2.3", "1.2.3-rc.1")).toBeGreaterThan(0);
		expect(comparePackageVersions("1.2.3-rc.1", "1.2.3-rc.2")).toBe(0);
		expect(comparePackageVersions("not-a-version", "1.2.3")).toBeUndefined();
	});
});

describe("checkAutoRAGUpdate", () => {
	it("reports an available update when the registry is newer", async () => {
		const fetchImpl = vi.fn(async () => jsonResponse("9.9.9"));
		const result = await checkAutoRAGUpdate({ currentVersion: "1.0.0", fetchImpl, env: {} });
		expect(result).toMatchObject({
			status: "available",
			currentVersion: "1.0.0",
			latestVersion: "9.9.9",
			packageName: AUTORAG_PACKAGE_NAME,
			installCommand: AUTORAG_UPDATE_COMMAND,
		});
		expect(fetchImpl).toHaveBeenCalledWith(AUTORAG_LATEST_VERSION_URL, expect.anything());
	});

	it("reports up-to-date when the registry matches or trails the running version", async () => {
		const same = await checkAutoRAGUpdate({
			currentVersion: "1.0.0",
			fetchImpl: vi.fn(async () => jsonResponse("1.0.0")),
			env: {},
		});
		expect(same.status).toBe("up-to-date");
		expect(same.latestVersion).toBe("1.0.0");
		const older = await checkAutoRAGUpdate({
			currentVersion: "2.0.0",
			fetchImpl: vi.fn(async () => jsonResponse("1.0.0")),
			env: {},
		});
		expect(older.status).toBe("up-to-date");
	});

	it("skips the lookup when AUTORAG_NO_UPDATE_CHECK is set", async () => {
		const fetchImpl = vi.fn(async () => jsonResponse("9.9.9"));
		const result = await checkAutoRAGUpdate({
			currentVersion: "1.0.0",
			fetchImpl,
			env: { AUTORAG_NO_UPDATE_CHECK: "1" },
		});
		expect(result.status).toBe("skipped");
		expect(fetchImpl).not.toHaveBeenCalled();
	});

	it("honors the AUTORAG_UPDATE_CHECK_URL override", async () => {
		const fetchImpl = vi.fn(async () => jsonResponse("9.9.9"));
		await checkAutoRAGUpdate({
			currentVersion: "1.0.0",
			fetchImpl,
			env: { AUTORAG_UPDATE_CHECK_URL: "http://localhost:9/latest" },
		});
		expect(fetchImpl).toHaveBeenCalledWith("http://localhost:9/latest", expect.anything());
	});

	it("never throws: transport, HTTP, and shape failures become error", async () => {
		const transport = await checkAutoRAGUpdate({
			currentVersion: "1.0.0",
			fetchImpl: vi.fn(async () => {
				throw new Error("offline");
			}),
			env: {},
		});
		expect(transport.status).toBe("error");

		const http = await checkAutoRAGUpdate({
			currentVersion: "1.0.0",
			fetchImpl: vi.fn(async () => new Response("nope", { status: 503 })),
			env: {},
		});
		expect(http.status).toBe("error");

		const shape = await checkAutoRAGUpdate({
			currentVersion: "1.0.0",
			fetchImpl: vi.fn(async () => new Response(JSON.stringify({}), { status: 200 })),
			env: {},
		});
		expect(shape.status).toBe("error");
	});
});

describe("renderAutoRAGUpdateNotice", () => {
	it("renders the version and install command only when an update is available", () => {
		const available: AutoRAGUpdateResult = {
			status: "available",
			packageName: AUTORAG_PACKAGE_NAME,
			currentVersion: "1.0.0",
			latestVersion: "2.0.0",
			installCommand: AUTORAG_UPDATE_COMMAND,
		};
		const notice = renderAutoRAGUpdateNotice(available);
		expect(notice).toContain("2.0.0");
		expect(notice).toContain("1.0.0");
		expect(notice).toContain(AUTORAG_UPDATE_COMMAND);
		expect(renderAutoRAGUpdateNotice({ ...available, status: "up-to-date" })).toBeUndefined();
		expect(renderAutoRAGUpdateNotice({ ...available, status: "error", latestVersion: undefined })).toBeUndefined();
	});
});
