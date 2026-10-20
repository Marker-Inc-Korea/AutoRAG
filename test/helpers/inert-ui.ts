import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";

/**
 * The interactive UI surface pi hands extensions, with no visible effect. `notify` records each
 * message into `notices`, so a test can read what an extension reported to the user.
 */
export function inertUi(notices: string[] = []): ExtensionUIContext {
	// A literal of the members extensions call; `theme` is the one member pi only supplies in a real TUI.
	const ui = {
		select: async () => undefined,
		confirm: async () => false,
		input: async () => undefined,
		notify: (message: string) => void notices.push(message),
		onTerminalInput: () => () => undefined,
		setStatus: () => undefined,
		setWorkingMessage: () => undefined,
		setWorkingVisible: () => undefined,
		setWorkingIndicator: () => undefined,
		setHiddenThinkingLabel: () => undefined,
		setWidget: () => undefined,
		setFooter: () => undefined,
		setHeader: () => undefined,
		setTitle: () => undefined,
		custom: async () => undefined,
		pasteToEditor: () => undefined,
		setEditorText: () => undefined,
		getEditorText: () => "",
		editor: async () => undefined,
		addAutocompleteProvider: () => undefined,
		setEditorComponent: () => undefined,
		getEditorComponent: () => undefined,
		theme: undefined,
		getAllThemes: () => [],
		getTheme: () => undefined,
		setTheme: () => ({ success: false, error: "UI not available" }),
		getToolsExpanded: () => false,
		setToolsExpanded: () => undefined,
	};
	return ui as unknown as ExtensionUIContext;
}
