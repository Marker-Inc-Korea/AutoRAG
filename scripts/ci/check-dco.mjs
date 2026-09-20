/**
 * DCO gate for pull requests (see CONTRIBUTING.md -> "Licensing and the DCO").
 *
 * Every commit in the pull-request range must carry a
 * `Signed-off-by: Name <email>` trailer whose email matches the commit's author or
 * committer email. Merge commits and bot authors are exempt.
 *
 * Usage:
 *   DCO_BASE_SHA=<base> DCO_HEAD_SHA=<head> node scripts/ci/check-dco.mjs
 *   node scripts/ci/check-dco.mjs <base> <head>
 *
 * Exit codes: 0 = every checked commit is signed off, 1 = at least one is not,
 * 2 = the check could not run (missing range, or git failed).
 */

import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

/** Authors whose name or email carries the GitHub app marker, e.g. `dependabot[bot]`. */
const BOT_AUTHOR = /\[bot\]/i;
/** A sign-off trailer: the line must start with the trailer token, per the DCO. */
const SIGN_OFF = /^\s*Signed-off-by:\s*(.*?)\s*<([^>\s]+)>\s*$/i;

/** Signers named by `Signed-off-by:` trailers in a commit message, lower-casing emails. */
export function parseSignOffs(message) {
  const signers = [];
  for (const line of String(message ?? "").split(/\r?\n/)) {
    const match = SIGN_OFF.exec(line);
    if (match) signers.push({ name: match[1], email: match[2].toLowerCase() });
  }
  return signers;
}

/** True when the commit author looks like a bot, which the DCO does not bind. */
export function isBotCommit(commit) {
  return BOT_AUTHOR.test(commit.authorName) || BOT_AUTHOR.test(commit.authorEmail);
}

/** Split a pull-request commit list into signed-off, offending, and exempt commits. */
export function evaluateCommits(commits) {
  const passed = [];
  const failed = [];
  const exempt = [];

  for (const commit of commits) {
    const parents = String(commit.parents ?? "").trim().split(/\s+/).filter(Boolean);
    if (parents.length > 1) {
      exempt.push({ commit, reason: "merge commit" });
      continue;
    }
    if (isBotCommit(commit)) {
      exempt.push({ commit, reason: `bot author (${commit.authorName})` });
      continue;
    }

    const signers = parseSignOffs(commit.message);
    if (signers.length === 0) {
      failed.push({ commit, reason: "no Signed-off-by: trailer" });
      continue;
    }

    // The DCO binds the person who submitted the commit; a rebase or cherry-pick can
    // make the committer that person, so either identity is accepted.
    const identities = new Set([
      String(commit.authorEmail ?? "").toLowerCase(),
      String(commit.committerEmail ?? "").toLowerCase(),
    ]);
    const signer = signers.find((entry) => identities.has(entry.email));
    if (signer) {
      passed.push({ commit, signer });
    } else {
      failed.push({
        commit,
        reason: `Signed-off-by email(s) ${signers.map((entry) => entry.email).join(", ")} do not match the author (${commit.authorEmail})`,
      });
    }
  }

  return { passed, failed, exempt };
}

function git(args, cwd) {
  const result = spawnSync("git", args, { cwd, encoding: "buffer", maxBuffer: 64 * 1024 * 1024 });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    const stderr = result.stderr ? result.stderr.toString("utf8").trim() : "";
    throw new Error(`git ${args.join(" ")} failed with exit status ${result.status}${stderr ? `: ${stderr}` : ""}`);
  }
  return result.stdout.toString("utf8");
}

// Fields are NUL-separated and the message is last, so a commit message can never be
// confused with the fields that precede it.
const COMMIT_FORMAT = "%an%x00%ae%x00%cn%x00%ce%x00%P%x00%B";

/** Read the commits in `range` (for example `<base>..<head>`) as DCO check records. */
export function readCommits(range, cwd = process.cwd()) {
  const shas = git(["log", "--format=%H", range, "--"], cwd)
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  return shas.map((sha) => {
    const parts = git(["show", "-s", `--format=${COMMIT_FORMAT}`, sha], cwd).split("\u0000");
    const [authorName, authorEmail, committerName, committerEmail, parents] = parts;
    return {
      sha,
      authorName,
      authorEmail,
      committerName,
      committerEmail,
      parents,
      message: parts.slice(5).join("\u0000").replace(/\n+$/, ""),
    };
  });
}

function subject(commit) {
  return String(commit.message ?? "").split("\n")[0].trim();
}

function main() {
  const argv = process.argv.slice(2);
  const base = argv[0] ?? process.env.DCO_BASE_SHA;
  const head = argv[1] ?? process.env.DCO_HEAD_SHA;
  if (!base || !head) {
    console.error(
      "check-dco: no commit range. Pass <base> <head>, or set DCO_BASE_SHA and DCO_HEAD_SHA.",
    );
    process.exit(2);
  }

  const range = `${base}..${head}`;
  let commits;
  try {
    commits = readCommits(range);
  } catch (error) {
    console.error(`check-dco: could not read ${range}\n${error instanceof Error ? error.message : String(error)}`);
    process.exit(2);
  }

  const { passed, failed, exempt } = evaluateCommits(commits);
  console.log(`check-dco: ${commits.length} commit(s) in ${range}`);

  for (const entry of passed) {
    console.log(`  ok      ${entry.commit.sha.slice(0, 8)}  ${entry.commit.authorName} <${entry.commit.authorEmail}>  ${subject(entry.commit)}`);
  }
  for (const entry of exempt) {
    console.log(`  exempt  ${entry.commit.sha.slice(0, 8)}  ${entry.reason}`);
  }

  if (failed.length === 0) {
    console.log(`check-dco: all ${passed.length} non-exempt commit(s) carry a matching Signed-off-by: trailer.`);
    process.exit(0);
  }

  console.error(`\ncheck-dco: ${failed.length} commit(s) are not signed off:`);
  for (const entry of failed) {
    console.error(`\n  ${entry.commit.sha}  ${entry.commit.authorName} <${entry.commit.authorEmail}>`);
    console.error(`    subject: ${subject(entry.commit)}`);
    console.error(`    reason:  ${entry.reason}`);
  }
  console.error(
    [
      "",
      "Every commit must certify the Developer Certificate of Origin:",
      "  new commits:   git commit -s",
      "  last commit:   git commit --amend -s",
      `  whole branch:  git rebase --signoff ${base}`,
      "then push with --force-with-lease. See CONTRIBUTING.md -> 'Licensing and the DCO'.",
      "",
    ].join("\n"),
  );
  process.exit(1);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
