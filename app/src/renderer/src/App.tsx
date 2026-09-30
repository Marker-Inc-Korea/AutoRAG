import { useCallback, useEffect, useState, type ReactElement } from "react";
import type { Permission } from "../../shared/settings-contract";
import type { SettingsBridge } from "../../shared/settings-contract";
import type { CitationEvidence } from "../../shared/search-contract";

const autorag = (window as unknown as { readonly autorag: { readonly settings: SettingsBridge } }).autorag;
import { AiSearchPanel } from "./components/AiSearchPanel";
import { EvidencePanel } from "./components/EvidencePanel";
import { ColumnHeader } from "./components/ColumnHeader";
import { FileList } from "./components/FileList";
import type { RowCallbacks } from "./components/FileRow";
import { ContextMenu } from "./components/primitives/ContextMenu";
import { Toast } from "./components/primitives/Toast";
import { Sidebar } from "./components/Sidebar";
import { SettingsPanel } from "./components/SettingsPanel";
import { RequestsPopover } from "./components/RequestsPopover";
import { StatusBar } from "./components/StatusBar";
import { formatDevLabel, type DevLabel } from "../../shared/app-info";
import { TabStrip } from "./components/TabStrip";
import { Toolbar } from "./components/Toolbar";
import { useFinderController } from "./hooks/useFinderController";
import { useFinderSource } from "./hooks/useFinderSource";

/** The reference's `showHotkeys` prop (default true): keycaps on or off. */
const SHOW_HOTKEYS = true;

/** Unpackaged runs label the window with their clone, branch, and commit. */
const DEV_LABEL = (() => {
	const dev = (window as unknown as { readonly autorag?: { readonly dev?: DevLabel | null } }).autorag?.dev;
	return dev === undefined || dev === null ? null : formatDevLabel(dev);
})();

export function App(): ReactElement {
	const source = useFinderSource();
	const [showHiddenFiles, setShowHiddenFiles] = useState(false);
	const finder = useFinderController(source, { showHiddenFiles });
	const [settingsTab, setSettingsTab] = useState<"general" | "contacts" | null>(null);
	const [requestsOpen, setRequestsOpen] = useState(false);
	const [pendingRequests, setPendingRequests] = useState(finder.pendingRequests);
	const [permissions, setPermissions] = useState<Readonly<Record<string, Permission>>>({});
	const [permissionTarget, setPermissionTarget] = useState<{ path: string; name: string } | null>(null);
	const [evidence, setEvidence] = useState<readonly CitationEvidence[]>([]);
	const [evidenceSession, setEvidenceSession] = useState<string | null>(null);
	const [selectedEvidence, setSelectedEvidence] = useState<number | null>(null);
	const [evidenceOpen, setEvidenceOpen] = useState(true);

	const publishEvidence = useCallback((items: readonly CitationEvidence[], sessionId: string | null) => {
		setEvidence(items);
		setEvidenceSession(sessionId);
		setSelectedEvidence((current) => (current !== null && items.some((item) => item.number === current) ? current : null));
	}, []);

	useEffect(() => {
		const onKeyDown = (event: globalThis.KeyboardEvent): void => {
			const target = event.target;
			if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return;
			if ((event.metaKey || event.ctrlKey) && !event.shiftKey && !event.altKey && event.key.toLowerCase() === "e") {
				event.preventDefault();
				setEvidenceOpen((open) => !open);
			}
		};
		window.addEventListener("keydown", onKeyDown);
		return () => window.removeEventListener("keydown", onKeyDown);
	}, []);

	const evidenceSelectedNumber = evidence.find((item) => item.number === selectedEvidence)?.number ?? evidence[0]?.number ?? 1;

	useEffect(() => {
		void autorag.settings.get().then((settings) => setShowHiddenFiles(settings.showHiddenFiles));
	}, []);

	useEffect(() => {
		const settingsBridge = autorag.settings;
		void settingsBridge.requestsList("pending").then((requests) => setPendingRequests(requests.length));
	}, [requestsOpen]);

	useEffect(() => {
		let active = true;
		void Promise.all(finder.rows.map(async (entry) => {
			const resolved = await autorag.settings.permGet(entry.path);
			return [entry.path, resolved.value] as const;
		})).then((values) => {
			if (active) setPermissions(Object.fromEntries(values));
		});
		return () => {
			active = false;
		};
	}, [finder.rows]);

	const rowCallbacks: RowCallbacks = {
		onClick: finder.clickRow,
		onOpen: finder.openEntry,
		onContextMenu: finder.openContextMenu,
		onToggleIndex: finder.toggleIndex,
		onToggleStack: finder.toggleStack,
		onChangeAccess: (entry) => setPermissionTarget({ path: entry.path, name: entry.name }),
		onRenameChange: finder.setRenameDraft,
		onRenameCommit: finder.commitRename,
		onRenameCancel: finder.cancelRename,
	};

	return (
		<div className="desk">
			<div className="window">
				<Sidebar
					activeLabel={finder.activeNavLabel}
					pendingRequests={pendingRequests}
					showHotkeys={SHOW_HOTKEYS}
					onNavigate={finder.navigateNav}
					onOpenRequests={() => setRequestsOpen(true)}
					onOpenSettings={() => setSettingsTab("general")}
				/>
				<section className="finder" aria-label="Finder" onMouseDown={finder.focusFinder}>
					<TabStrip
						tabs={finder.tabs}
						canClose={finder.canCloseTabs}
						onSelect={finder.selectTabId}
						onClose={finder.closeTabId}
						onNewTab={finder.newTab}
					/>
					<Toolbar
						crumbs={finder.crumbs}
						canGoBack={finder.canGoBack}
						canGoForward={finder.canGoForward}
						query={finder.query}
						placeholder={finder.searchPlaceholder}
						searchFocused={finder.searchFocused}
						showHotkeys={SHOW_HOTKEYS}
						searchRef={finder.searchRef}
						onNavigate={finder.navigate}
						onBack={finder.back}
						onForward={finder.forward}
						onQueryChange={finder.setQuery}
						onSearchKeyDown={finder.onSearchKeyDown}
						onSearchFocus={() => finder.setSearchFocused(true)}
						onSearchBlur={() => finder.setSearchFocused(false)}
					/>
					<ColumnHeader sort={finder.sort} searching={finder.searching} onSort={finder.sortBy} />
					<FileList
						rows={finder.stackRows}
						listPath={finder.path}
						versionFamilyError={finder.versionFamilyError}
						onRetryVersionFamilies={finder.retryVersionFamilies}
						listRef={finder.listRef}
						searching={finder.searching}
						summary={finder.searchSummary}
						emptyText={finder.emptyText}
						selectedKeys={finder.selection.keys}
						focusedZone={finder.zone === "finder"}
						flashPath={finder.flashPath}
						renamePath={finder.renamePath}
						renameDraft={finder.renameDraft}
						indexOverrides={finder.indexOverrides}
						permissions={permissions}
						callbacks={rowCallbacks}
					/>
					<StatusBar text={finder.statusText} showHotkeys={SHOW_HOTKEYS} devLabel={DEV_LABEL} />
					{evidence.length === 0 ? null : (
						<EvidencePanel
							evidence={evidence}
							sessionId={evidenceSession}
							selected={evidenceSelectedNumber}
							open={evidenceOpen}
							onToggleOpen={() => setEvidenceOpen((open) => !open)}
							onSelect={(number) => {
								setSelectedEvidence(number);
								setEvidenceOpen(true);
							}}
							onQuickLook={finder.quickLookPath}
							onReveal={finder.revealEvidence}
							onToast={finder.showToast}
						/>
					)}
				</section>
				<AiSearchPanel selectedEvidence={evidence.length === 0 ? null : evidenceSelectedNumber} onSelectEvidence={(number) => { setSelectedEvidence(number); setEvidenceOpen(true); }} onPublish={publishEvidence} />
				{finder.toast === null ? null : <Toast message={finder.toast} />}
			</div>
			{finder.contextMenu === null ? null : (
				<ContextMenu
					cursor={finder.contextMenu.cursor}
					entries={finder.contextMenuEntries}
					onSelect={finder.runMenuAction}
					onClose={finder.closeContextMenu}
				/>
			)}
			{settingsTab === null ? null : (
				<SettingsPanel
					initialTab={settingsTab}
					onClose={() => setSettingsTab(null)}
					onSettingsChanged={(settings) => setShowHiddenFiles(settings.showHiddenFiles)}
				/>
			)}
			{requestsOpen ? <RequestsPopover onClose={() => setRequestsOpen(false)} onChanged={() => void autorag.settings.requestsList("pending").then((requests) => setPendingRequests(requests.length))} /> : null}
			{permissionTarget === null ? null : (
				<div className="settings-backdrop" role="dialog" aria-modal="true" aria-label="Access permission">
					<section className="settings-panel">
						<header className="settings-panel__header">
							<strong>Access: {permissionTarget.name}</strong>
							<button type="button" onClick={() => setPermissionTarget(null)}>Close</button>
						</header>
						<div className="settings-panel__body">
							<p>Choose who may access this file or folder. Folder changes apply to its descendants.</p>
							{(["ask", "allow", "deny"] as const).map((value) => (
								<button
									type="button"
									key={value}
									onClick={() => {
										void autorag.settings.permSet(permissionTarget.path, value, "keep-individual").then(() => {
											setPermissions((current) => ({ ...current, [permissionTarget.path]: value }));
											setPermissionTarget(null);
										});
									}}
								>{value[0]?.toUpperCase() + value.slice(1)}</button>
							))}
						</div>
					</section>
				</div>
			)}
		</div>
	);
}
