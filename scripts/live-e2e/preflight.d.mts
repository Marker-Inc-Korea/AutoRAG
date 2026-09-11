export function runPreflight(options?: {
  readonly endpoint?: string;
  readonly fetchImpl?: (input: string, init?: object) => Promise<{ readonly ok: boolean; readonly status: number; readonly json: () => Promise<unknown> }>;
  readonly binaries?: Readonly<Record<string, boolean>>;
  readonly configuredLanes?: readonly string[];
  readonly openaiKey?: string;
}): Promise<{ readonly verdict: string; readonly exitCode: number; readonly code?: string; readonly openaiKeyPresent: boolean; readonly embedding: { readonly status: string; readonly dimension?: number }; readonly lanes: Record<string, { readonly status: string; readonly reason?: string }> }>;
