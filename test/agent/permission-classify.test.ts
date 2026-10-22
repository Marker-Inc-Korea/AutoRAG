import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
	type ClassifyFs,
	classifyAutomationRun,
	classifyBash,
	classifyFileTool,
	type PermissionOperation,
} from "../../src/agent/permission-classify.ts";

const CWD = resolve("/work");
const p = (...parts: string[]): string => resolve(CWD, ...parts);

/** Files and directories that "exist" for a classification. */
function fsOf(files: readonly string[], dirs: readonly string[] = [], home?: string): ClassifyFs {
	const fileSet = new Set(files.map((f) => resolve(f)));
	const dirSet = new Set(dirs.map((d) => resolve(d)));
	return {
		cwd: CWD,
		...(home === undefined ? {} : { home }),
		exists: (path: string) => fileSet.has(path) || dirSet.has(path),
		isDirectory: (path: string) => dirSet.has(path),
	};
}

/** The gated (non-create) operations of one command, as `kind:path` strings. */
function gated(command: string, fs = fsOf([])): string[] {
	return classifyBash(command, fs)
		.filter((op) => op.kind !== "create")
		.flatMap((op) => op.paths.map((path) => `${op.kind}:${path}`));
}

describe("classifyBash", () => {
	it("passes read-only and copy commands", () => {
		const fs = fsOf([p("a.pdf")]);
		expect(classifyBash("ls -la docs && grep -rn foo docs | head", fs)).toEqual([]);
		expect(classifyBash("cat a.pdf", fs)).toEqual([]);
		expect(gated("cp a.pdf b.pdf", fs)).toEqual([]);
		expect(gated("mkdir -p out/sub && touch out/x", fs)).toEqual([]);
	});

	it("flags rm, rmdir, unlink, and trash as delete", () => {
		expect(gated("rm -rf old a.txt")).toEqual([`delete:${p("old")}`, `delete:${p("a.txt")}`]);
		expect(gated("rmdir empty")).toEqual([`delete:${p("empty")}`]);
		expect(gated("unlink x")).toEqual([`delete:${p("x")}`]);
		expect(gated("trash a b")).toEqual([`delete:${p("a")}`, `delete:${p("b")}`]);
		expect(gated("/bin/rm -- -weird")).toEqual([`delete:${p("-weird")}`]);
	});

	it("keeps quoted paths with spaces whole", () => {
		expect(gated(`rm "my file.pdf" 'other file.pdf'`)).toEqual([
			`delete:${p("my file.pdf")}`,
			`delete:${p("other file.pdf")}`,
		]);
	});

	it("tells rename from move", () => {
		const fs = fsOf([p("a.pdf")], [p("archive")]);
		expect(gated("mv a.pdf b.pdf", fs)).toEqual([`rename:${p("a.pdf")}`]);
		expect(gated("mv a.pdf archive", fs)).toEqual([`move:${p("a.pdf")}`]);
		expect(gated("mv a.pdf archive/a.pdf", fs)).toEqual([`move:${p("a.pdf")}`]);
		expect(gated("mv -t archive a.pdf", fs)).toEqual([`move:${p("a.pdf")}`]);
		expect(gated("mv a.pdf sub/b.pdf", fs)).toEqual([`move:${p("a.pdf")}`]);
	});

	it("flags overwriting copies, redirects, tee, and in-place editors as modifying an original", () => {
		const fs = fsOf([p("a.txt"), p("b.txt")]);
		expect(gated("cp a.txt b.txt", fs)).toEqual([`overwrite:${p("b.txt")}`]);
		expect(gated("cp -n a.txt b.txt", fs)).toEqual([]);
		expect(gated("echo hi > a.txt", fs)).toEqual([`overwrite:${p("a.txt")}`]);
		expect(gated("echo hi >> a.txt", fs)).toEqual([`overwrite:${p("a.txt")}`]);
		expect(gated("echo hi > new.txt", fs)).toEqual([]);
		expect(gated("echo hi > /dev/null 2>&1", fs)).toEqual([]);
		expect(gated("echo hi | tee a.txt", fs)).toEqual([`overwrite:${p("a.txt")}`]);
		expect(gated("sed -i 's/a/b/' a.txt", fs)).toEqual([`edit:${p("a.txt")}`]);
		expect(gated("sed -i.bak -e 's/a/b/' a.txt b.txt", fs)).toEqual([`edit:${p("a.txt")}`, `edit:${p("b.txt")}`]);
		expect(gated("sed 's/a/b/' a.txt", fs)).toEqual([]);
		expect(gated("perl -pi -e 's/a/b/' a.txt", fs)).toEqual([`edit:${p("a.txt")}`]);
		expect(gated("truncate -s 0 a.txt", fs)).toEqual([`edit:${p("a.txt")}`]);
	});

	it("treats every OS trash command as a delete", () => {
		expect(gated("/usr/bin/trash a.pdf")).toEqual([`delete:${p("a.pdf")}`]);
		expect(gated("gio trash a.pdf b.pdf")).toEqual([`delete:${p("a.pdf")}`, `delete:${p("b.pdf")}`]);
		expect(gated("trash-put a.pdf")).toEqual([`delete:${p("a.pdf")}`]);
		expect(gated("gio list .")).toEqual([]);
		expect(gated("mv a.pdf ~/.Trash/", fsOf([p("a.pdf")], [], resolve("/home/me")))).toEqual([`move:${p("a.pdf")}`]);
		expect(
			classifyBash(`osascript -e 'tell application "Finder" to delete POSIX file "/tmp/x"'`, fsOf([])).map(
				(op) => op.kind,
			),
		).toEqual(["script"]);
	});

	it("reports the backup file `sed -i.bak` writes as a creation", () => {
		const ops = classifyBash("sed -i.bak 's/a/b/' a.txt", fsOf([p("a.txt")]));
		expect(ops.filter((op) => op.kind === "create").flatMap((op) => op.paths)).toEqual([p("a.txt.bak")]);
		expect(classifyBash("sed -i 's/a/b/' a.txt", fsOf([p("a.txt")])).filter((op) => op.kind === "create")).toEqual(
			[],
		);
		expect(
			classifyBash("sed --in-place=.orig -e 's/a/b/' a.txt", fsOf([p("a.txt")]))
				.filter((op) => op.kind === "create")
				.flatMap((op) => op.paths),
		).toEqual([p("a.txt.orig")]);
	});

	it("looks inside a heredoc redirect without parsing the body as commands", () => {
		const fs = fsOf([p("a.txt")]);
		expect(gated("cat <<'EOF' > a.txt\nrm -rf / ; it's fine\nEOF", fs)).toEqual([`overwrite:${p("a.txt")}`]);
		expect(gated("cat <<EOF > fresh.txt\nrm -rf /\nEOF", fs)).toEqual([]);
	});

	it("follows chains, pipes, subshells, and cd", () => {
		expect(gated("cd docs && rm a.pdf; mv b.pdf c.pdf")).toEqual([
			`delete:${p("docs", "a.pdf")}`,
			`rename:${p("docs", "b.pdf")}`,
		]);
		expect(classifyBash("ls | xargs rm", fsOf([]))).toMatchObject([{ kind: "delete", dynamic: true }]);
		expect(gated("(rm a)")).toEqual([`delete:${p("a")}`]);
		expect(gated("echo $(rm a)")).toEqual([`delete:${p("a")}`]);
		expect(gated("echo `rm a`")).toEqual([`delete:${p("a")}`]);
	});

	it("looks through wrappers and nested shells", () => {
		expect(gated("sudo rm a")).toEqual([`delete:${p("a")}`]);
		expect(gated("env FOO=1 nohup rm a")).toEqual([`delete:${p("a")}`]);
		expect(gated("FOO=1 rm a")).toEqual([`delete:${p("a")}`]);
		expect(gated(`bash -c "rm a && mv b c"`)).toEqual([`delete:${p("a")}`, `rename:${p("b")}`]);
		expect(gated(`eval "rm a"`)).toEqual([`delete:${p("a")}`]);
	});

	it("flags find -delete, -exec rm, rsync removal, and git rm/clean", () => {
		expect(gated("find old -name '*.tmp' -delete")).toEqual([`delete:${p("old")}`]);
		expect(gated("find . -type f -exec rm {} +")).toEqual([`delete:${p(".")}`]);
		expect(gated("find . -name x -print")).toEqual([]);
		expect(gated("rsync -a --remove-source-files src/ dst/")).toEqual([`delete:${p("src")}`]);
		expect(gated("git rm a.txt")).toEqual([`delete:${p("a.txt")}`]);
		expect(gated("git mv a b")).toEqual([`rename:${p("a")}`]);
		expect(gated("git clean -fd")).toEqual([`delete:${CWD}`]);
		expect(gated("git status && git log")).toEqual([]);
	});

	it("marks paths it cannot resolve as dynamic", () => {
		const ops = classifyBash("rm $TARGET", fsOf([]));
		expect(ops).toHaveLength(1);
		expect(ops[0]).toMatchObject({ kind: "delete", dynamic: true });
		expect(classifyBash("rm *.tmp", fsOf([]))[0]).toMatchObject({ kind: "delete", dynamic: true });
	});

	it("flags interpreter one-liners that delete or move files", () => {
		const kinds = (command: string): string[] => classifyBash(command, fsOf([])).map((op) => op.kind);
		expect(kinds(`python3 -c "import os; os.remove('a')"`)).toEqual(["script"]);
		expect(kinds(`node -e "require('fs').rmSync('a')"`)).toEqual(["script"]);
		expect(kinds("python3 - <<'PY'\nimport shutil\nshutil.move('a','b')\nPY")).toEqual(["script"]);
		expect(kinds(`python3 -c "print(1)"`)).toEqual([]);
	});

	it("reports each creation so the gate can trust files the agent made", () => {
		const ops = classifyBash("cp a.pdf copy.pdf && mkdir out && echo x > out/n.txt", fsOf([p("a.pdf")]));
		const created = ops.filter((op) => op.kind === "create").flatMap((op) => op.paths);
		expect(created).toEqual([p("copy.pdf"), p("out"), p("out", "n.txt")]);
	});
});

describe("classifyFileTool", () => {
	it("treats write to a missing path as a creation", () => {
		const ops = classifyFileTool("write", { path: "new.txt" }, fsOf([]));
		expect(ops).toEqual<PermissionOperation[]>([
			{ kind: "create", paths: [p("new.txt")], detail: `create ${p("new.txt")}` },
		]);
	});

	it("treats write over an existing file as overwrite and edit as edit", () => {
		const fs = fsOf([p("a.txt")]);
		expect(classifyFileTool("write", { path: "a.txt" }, fs)).toMatchObject([
			{ kind: "overwrite", paths: [p("a.txt")] },
		]);
		expect(classifyFileTool("edit", { path: "a.txt" }, fs)).toMatchObject([{ kind: "edit", paths: [p("a.txt")] }]);
	});

	it("expands ~ and tolerates a missing path argument", () => {
		const home = resolve("/home/me");
		const fs = { ...fsOf([resolve(home, "n.md")]), home };
		expect(classifyFileTool("edit", { path: "~/n.md" }, fs)).toMatchObject([{ paths: [resolve(home, "n.md")] }]);
		expect(classifyFileTool("edit", {}, fs)).toEqual([]);
	});
});

describe("classifyAutomationRun", () => {
	it("gates a saved automation whose command deletes or moves files", () => {
		const kinds = (run: string[]) => classifyAutomationRun(run, fsOf([])).map((op) => op.kind);
		expect(kinds(["rm", "-f", "x"])).toEqual(["delete"]);
		expect(kinds(["sh", "-c", "mv a b"])).toEqual(["rename"]);
		expect(kinds(["autorag", "search", "summarize"])).toEqual([]);
	});
});
