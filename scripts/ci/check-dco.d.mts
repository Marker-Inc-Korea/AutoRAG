export interface DcoCommit {
  readonly sha: string;
  readonly authorName: string;
  readonly authorEmail: string;
  readonly committerName: string;
  readonly committerEmail: string;
  readonly parents: string;
  readonly message: string;
}

export interface DcoSigner {
  readonly name: string;
  readonly email: string;
}

export interface DcoEvaluation {
  readonly passed: ReadonlyArray<{ readonly commit: DcoCommit; readonly signer: DcoSigner }>;
  readonly failed: ReadonlyArray<{ readonly commit: DcoCommit; readonly reason: string }>;
  readonly exempt: ReadonlyArray<{ readonly commit: DcoCommit; readonly reason: string }>;
}

export function parseSignOffs(message: string): DcoSigner[];
export function isBotCommit(commit: DcoCommit): boolean;
export function evaluateCommits(commits: readonly DcoCommit[]): DcoEvaluation;
export function readCommits(range: string, cwd?: string): DcoCommit[];
