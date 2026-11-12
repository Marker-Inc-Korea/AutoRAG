/**
 * Built-in channel kinds. Each kind lives in its own file and registers itself with {@link registerChannelKind};
 * this is the only place that lists them. Adding a channel is one file plus one line here.
 */

let registered = false;

export function registerBuiltinChannelKinds(): void {
	if (registered) return;
	registered = true;
}
