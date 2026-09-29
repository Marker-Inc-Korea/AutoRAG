import { useEffect, useRef, useState, type FormEvent, type ReactElement } from "react";
import type { AppSettings, AssistantEvent, Contact, DataSourceRow } from "../../../shared/settings-contract";
import type { SettingsBridge, AssistantSurface } from "../../../shared/settings-contract";
import { ArrowUpIcon, CloseIcon, SparklesIcon } from "./icons";

const autorag = (window as unknown as { readonly autorag: { readonly settings: SettingsBridge } }).autorag;

type SettingsTab = "general" | "models" | "sources" | "contacts";

/** Reference greeting copy (proto cfgMsgs / srcMsgs initial assistant messages, verbatim). */
const GREETINGS: Record<AssistantSurface, string> = {
	models: "모델 설정을 도와드릴게요. 원하는 방식을 말씀해 주세요. Quick과 Deep 답변은 같은 모델을 사용합니다.",
	sources: "데이터 소스를 연결하거나, 에이전트가 읽는 소스 설명을 다듬어 드릴게요.",
};

const CHIPS: Record<AssistantSurface, readonly string[]> = {
	models: ["가장 정확한 모델로", "빠른 응답 위주로", "데이터가 기기 밖으로 안 나가게"],
	sources: ["연결 가능한 데이터소스들 나열", "데이터소스 설명 손보기", "Discord 다시 연결"],
};

const PLACEHOLDERS: Record<AssistantSurface, string> = {
	models: "예: 회사 데이터는 밖으로 안 나가게 해줘",
	sources: "예: Outlook 연결해줘, Slack 설명 손봐줘",
};

interface ChatMessage {
	readonly role: "user" | "assistant";
	readonly text: string;
}

/** Source-kind tile colors reuse the v6 file/datasource tile tokens (DESIGN.md §2.9). */
function kindTileModifier(kind: string): string {
	switch (kind) {
		case "slack":
		case "slacrawl":
			return "slack";
		case "discrawl":
		case "discord":
			return "discord";
		case "telecrawl":
		case "telegram":
			return "telegram";
		case "mailcrawl":
		case "mail-export":
		case "gmail":
			return "mail";
		case "notcrawl":
		case "notion":
			return "notion";
		default:
			return "generic";
	}
}

function statusModifier(status: DataSourceRow["status"]): string {
	return `source-status--${status}`;
}

export function SettingsPanel({
	initialTab = "general",
	onClose,
	onSettingsChanged,
}: {
	readonly initialTab?: SettingsTab;
	readonly onClose: () => void;
	readonly onSettingsChanged?: (settings: AppSettings) => void;
}): ReactElement {
	const [tab, setTab] = useState<SettingsTab>(initialTab);
	const [settings, setSettings] = useState<AppSettings | null>(null);
	const [contacts, setContacts] = useState<readonly Contact[]>([]);
	const [sources, setSources] = useState<readonly DataSourceRow[]>([]);
	const [openSource, setOpenSource] = useState<string | null>(null);
	const [contactDraft, setContactDraft] = useState({ id: "", name: "", role: "", description: "" });
	const [chats, setChats] = useState<Record<AssistantSurface, ChatMessage[]>>({
		models: [{ role: "assistant", text: GREETINGS.models }],
		sources: [{ role: "assistant", text: GREETINGS.sources }],
	});
	const [drafts, setDrafts] = useState<Record<AssistantSurface, string>>({ models: "", sources: "" });
	const [busy, setBusy] = useState(false);
	const [updateState, setUpdateState] = useState<"idle" | "checking" | "done">("idle");
	const chatListRef = useRef<HTMLDivElement | null>(null);
	const chatInputRef = useRef<HTMLInputElement | null>(null);

	useEffect(() => {
		void Promise.all([
			autorag.settings.get().then(setSettings),
			autorag.settings.contactsList().then(setContacts),
			autorag.settings.sourcesList().then(setSources),
		]);
		const unsubscribe = autorag.settings.onEvent?.((event: AssistantEvent) => {
			if (event.type === "text") {
				setChats((current) => {
					const thread = current[event.surface];
					const last = thread[thread.length - 1];
					const nextThread =
						last !== undefined && last.role === "assistant"
							? [...thread.slice(0, -1), { role: "assistant" as const, text: last.text + event.delta }]
							: [...thread, { role: "assistant" as const, text: event.delta }];
					return { ...current, [event.surface]: nextThread };
				});
			}
			if (event.type === "done" || event.type === "error") setBusy(false);
			if (event.type === "error") {
				setChats((current) => ({
					...current,
					[event.surface]: [...current[event.surface], { role: "assistant" as const, text: event.message }],
				}));
			}
		});
		return unsubscribe;
	}, []);

	useEffect(() => {
		const list = chatListRef.current;
		if (list !== null) list.scrollTop = list.scrollHeight;
	}, [chats, tab]);

	async function sendAssistant(surface: AssistantSurface, raw: string): Promise<void> {
		const text = raw.trim();
		if (text.length === 0 || busy) return;
		setBusy(true);
		setChats((current) => ({ ...current, [surface]: [...current[surface], { role: "user", text }] }));
		setDrafts((current) => ({ ...current, [surface]: "" }));
		await autorag.settings.assistantSend(surface, text);
	}

	async function toggleSetting(key: keyof AppSettings): Promise<void> {
		if (settings === null || typeof settings[key] !== "boolean") return;
		const next = await autorag.settings.set({ [key]: !settings[key] });
		setSettings(next);
		onSettingsChanged?.(next);
	}

	async function addContact(event: FormEvent<HTMLFormElement>): Promise<void> {
		event.preventDefault();
		if (Object.values(contactDraft).some((value) => value.trim().length === 0)) return;
		const result = await autorag.settings.contactsAdd(contactDraft);
		if (result.ok) {
			setContacts((items) => [...items, contactDraft]);
			setContactDraft({ id: "", name: "", role: "", description: "" });
		}
	}

	async function toggleSource(source: DataSourceRow): Promise<void> {
		await autorag.settings.sourcesSetEnabled(source.id, !source.enabled);
		setSources((items) => items.map((item) => item.id === source.id ? { ...item, enabled: !item.enabled } : item));
	}

	function editSourceDescription(source: DataSourceRow): void {
		setTab("sources");
		setDrafts((current) => ({ ...current, sources: `${source.name} 설명 손보기: ` }));
		setTimeout(() => chatInputRef.current?.focus(), 0);
	}

	// WIP: per-source indexed item counts and last-sync timestamps are not
	// tracked yet — the summary reports source counts only.
	const syncingCount = sources.filter((s) => s.enabled && s.status === "syncing").length;
	const attentionCount = sources.filter((s) => s.enabled && s.status === "error").length;
	const sourcesSummary = [
		`${sources.length} ${sources.length === 1 ? "source" : "sources"}`,
		syncingCount > 0 ? `${syncingCount} syncing` : null,
		attentionCount > 0 ? `${attentionCount} needs attention` : null,
	].filter(Boolean).join(" · ");

	return (
		// biome-ignore lint/a11y/useKeyWithClickEvents: modal backdrop dismiss mirrors the prototype
		<div className="settings-backdrop" role="dialog" aria-modal="true" aria-label="Settings" onClick={onClose}>
			{/* biome-ignore lint/a11y/noStaticElementInteractions: clicks inside the panel must not close it */}
			<section className="settings-panel" onClick={(event) => event.stopPropagation()}>
				<header className="settings-panel__header">
					<strong className="settings-panel__title">Settings</strong>
					<nav className="settings-panel__tabs" aria-label="Settings sections">
						{(["general", "models", "sources"] as const).map((item) => (
							<button
								type="button"
								className={tab === item ? "is-active" : ""}
								key={item}
								onClick={() => setTab(item)}
							>
								{item === "sources" ? "Data Sources" : item[0]?.toUpperCase() + item.slice(1)}
							</button>
						))}
					</nav>
					<div className="settings-panel__spacer" />
					<button type="button" className="icon-button" aria-label="Close Settings" title="Close" onClick={onClose}><CloseIcon /></button>
				</header>
				<div className="settings-panel__body">
					{tab === "general" && settings !== null ? (
						<>
							<SettingsSection title="일반">
								<SettingRow label="Language" description="앱 인터페이스 언어">
									<div className="settings-select"><select value={settings.language} onChange={(event) => void autorag.settings.set({ language: event.target.value as AppSettings["language"] }).then(setSettings)}>
										<option value="ko">한국어</option><option value="en">English</option><option value="ja">日本語</option><option value="zh">简体中文</option>
									</select></div>
								</SettingRow>
								{/* WIP: the global shortcut is shown but main-process registration is not implemented. */}
								<SettingRow label="Global shortcut" description="어디서든 AutoRAG Agent 열기">
									<div className="settings-keycaps"><kbd>⌥</kbd><kbd>Space</kbd></div>
								</SettingRow>
								{/* WIP: launch-at-login persists but the app does not call app.setLoginItemSettings yet. */}
								<SettingRow label="Launch at login" description="Mac 로그인 시 자동 실행">
									<Switch checked={settings.launchAtLogin} onChange={() => void toggleSetting("launchAtLogin")} />
								</SettingRow>
								{/* WIP: menu-bar tray visibility is persisted but no Tray is managed in main yet. */}
								<SettingRow label="Show in menu bar" description="메뉴 막대에 아이콘 표시">
									<Switch checked={settings.showInMenuBar} onChange={() => void toggleSetting("showInMenuBar")} />
								</SettingRow>
								<SettingRow label="Show hidden files" description="점(.)으로 시작하는 숨김 파일·폴더를 목록에 흐리게 표시">
									<Switch checked={settings.showHiddenFiles} onChange={() => void toggleSetting("showHiddenFiles")} />
								</SettingRow>
							</SettingsSection>
							{/* WIP: there is no real updater; the check below is simulated UI only. */}
							<SettingsSection title="소프트웨어 업데이트">
								<SettingRow
									label="AutoRAG Agent 1.4.2"
									description={updateState === "done" ? "최신 버전입니다 · 방금 확인" : "마지막 확인: 오늘 09:00"}
								>
									<button type="button" onClick={() => { setUpdateState("checking"); window.setTimeout(() => setUpdateState("done"), 500); }}>{updateState === "checking" ? "Checking…" : "Check for Updates"}</button>
								</SettingRow>
								{/* WIP: auto-install persists; no update download/apply pipeline exists yet. */}
								<SettingRow label="Automatically install updates" description="백그라운드에서 다운로드 후 재시작 시 적용">
									<Switch checked={settings.autoInstallUpdates} onChange={() => void toggleSetting("autoInstallUpdates")} />
								</SettingRow>
							</SettingsSection>
							<SettingsSection title="개인정보">
								{/* WIP: settings-bridge clearHistory is a stub; the chat store is not cleared yet. */}
								<SettingRow label="Clear chat history" description="저장된 모든 AI Search 대화를 삭제">
									<button type="button" className="settings-danger-button" onClick={() => void autorag.settings.clearHistory()}>Clear…</button>
								</SettingRow>
								{/* WIP: telemetry persists but no diagnostics pipeline reads it. */}
								<SettingRow label="Share anonymous diagnostics" description="제품 개선을 위한 익명 진단 정보 공유">
									<Switch checked={settings.telemetry} onChange={() => void toggleSetting("telemetry")} />
								</SettingRow>
							</SettingsSection>
						</>
					) : null}
					{tab === "contacts" ? (
						<>
							<form className="settings-card settings-contact-form" onSubmit={(event) => void addContact(event)}>
								<input placeholder="id" value={contactDraft.id} onChange={(event) => setContactDraft({ ...contactDraft, id: event.target.value })} />
								<input placeholder="name" value={contactDraft.name} onChange={(event) => setContactDraft({ ...contactDraft, name: event.target.value })} />
								<input placeholder="role" value={contactDraft.role} onChange={(event) => setContactDraft({ ...contactDraft, role: event.target.value })} />
								<input placeholder="description" value={contactDraft.description} onChange={(event) => setContactDraft({ ...contactDraft, description: event.target.value })} />
								<button type="submit">Add contact</button>
							</form>
							{contacts.length === 0 ? <p>No contacts yet.</p> : contacts.map((contact) => (
								<article className="settings-card" key={contact.id}>
									<strong>{contact.name}</strong><span>{contact.role}</span><p>{contact.description}</p>
									<button type="button" onClick={() => void autorag.settings.contactsRemove(contact.id).then(() => setContacts((items) => items.filter((item) => item.id !== contact.id)))}>Remove</button>
								</article>
							))}
						</>
					) : null}
					{tab === "sources" ? (
						<div className="settings-stack">
							<div className="settings-intro">
								<strong className="settings-intro__title">Connected sources</strong>
								<span className="settings-intro__desc">{sourcesSummary}</span>
							</div>
							<div className="settings-sources">
								{sources.length === 0 ? <p className="settings-sources__empty">No configured data sources.</p> : sources.map((source) => (
									<SourceRow
										key={source.id}
										source={source}
										open={openSource === source.id}
										onToggle={() => void toggleSource(source)}
										onDetail={() => setOpenSource((current) => (current === source.id ? null : source.id))}
										onEditDescription={() => editSourceDescription(source)}
									/>
								))}
							</div>
							<AssistantPane surface="sources" chats={chats.sources} draft={drafts.sources} busy={busy} listRef={chatListRef} inputRef={chatInputRef} setDraft={(text) => setDrafts((current) => ({ ...current, sources: text }))} onSend={sendAssistant} />

						</div>
					) : null}
					{tab === "models" ? (
						<div className="settings-stack">
							{/* WIP: the bridge does not expose the configured model identity; the card is static. */}
							<div className="settings-model-card">
								<span className="settings-model-tile">A</span>
								<div>
									<strong className="settings-model-card__name">AutoRAG Agent</strong>
									<span className="settings-model-card__provider">Connected model</span>
								</div>
								<em>Quick · Deep 공통</em>
							</div>
							<AssistantPane surface="models" chats={chats.models} draft={drafts.models} busy={busy} listRef={chatListRef} inputRef={null} setDraft={(text) => setDrafts((current) => ({ ...current, models: text }))} onSend={sendAssistant} />
						</div>
					) : null}
				</div>
			</section>
		</div>
	);
}

function SourceRow({
	source,
	open,
	onToggle,
	onDetail,
	onEditDescription,
}: {
	readonly source: DataSourceRow;
	readonly open: boolean;
	readonly onToggle: () => void;
	readonly onDetail: () => void;
	readonly onEditDescription: () => void;
}): ReactElement {
	const status: DataSourceRow["status"] = source.enabled ? source.status : "paused";
	const progress = source.progress ?? 0;
	return (
		<div className={`settings-source${open ? " is-open" : ""}`}>
			<div className={`settings-source__row${source.enabled ? "" : " is-disabled"}`}>
				<span className={`settings-source-tile settings-source-tile--${kindTileModifier(source.kind)}`}>{source.name[0] ?? "?"}</span>
				{/* biome-ignore lint/a11y/noStaticElementInteractions: row body mirrors the prototype's clickable detail target */}
				<div className="settings-source__body" onClick={onDetail}>
					<span className="settings-source__name">{source.name}</span>
					<span className="settings-source__detail">{source.detail}</span>
				</div>
				<div className="settings-source__track">
					<div className="settings-source__meta">
						<span className={`settings-status ${statusModifier(status)}`}>{status[0]?.toUpperCase() + status.slice(1)}</span>
						<span className="settings-source__pct">{source.progress === null ? "" : `${Math.round(progress)}%`}</span>
					</div>
					<div className="settings-progress"><div className={`settings-progress__fill settings-progress__fill--${status}`} style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} /></div>
				</div>
				<button type="button" className={`settings-source__chevron${open ? " is-open" : ""}`} title="Description" onClick={onDetail}>
					<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 9l6 6 6-6" /></svg>
				</button>
				<Switch checked={source.enabled} onChange={onToggle} />
			</div>
			{open ? (
				<div className="settings-source-desc">
					<div className="settings-source-desc__header">
						<span>Description</span>
						<div className="settings-source-desc__spacer" />
						<button type="button" onClick={onEditDescription}>채팅으로 수정</button>
					</div>
					<p>{source.description}</p>
				</div>
			) : null}
		</div>
	);
}

function AssistantPane({
	surface,
	chats,
	draft,
	busy,
	listRef,
	inputRef,
	setDraft,
	onSend,
}: {
	readonly surface: AssistantSurface;
	readonly chats: readonly ChatMessage[];
	readonly draft: string;
	readonly busy: boolean;
	readonly listRef: React.RefObject<HTMLDivElement | null> | null;
	readonly inputRef: React.RefObject<HTMLInputElement | null> | null;
	readonly setDraft: (text: string) => void;
	readonly onSend: (surface: AssistantSurface, text: string) => Promise<void>;
}): ReactElement {
	return (
		<div className={`settings-chat settings-chat--${surface}`}>
			<div className="settings-chat__messages" ref={listRef ?? undefined}>
				{chats.map((message, index) => (
					<div className={`settings-chat__line settings-chat__line--${message.role}`} key={index}>
						<div className={`settings-chat__bubble settings-chat__bubble--${message.role}`}>{message.text}</div>
					</div>
				))}
			</div>
			<div className="settings-chips">
				{CHIPS[surface].map((chip) => (
					<button type="button" key={chip} disabled={busy} onClick={() => void onSend(surface, chip)}>{chip}</button>
				))}
			</div>
			<form
				className="settings-chat__composer"
				onSubmit={(event) => {
					event.preventDefault();
					void onSend(surface, draft);
				}}
			>
				<SparklesIcon size={14} className="settings-chat__sparkle" />
				<input
					ref={inputRef ?? undefined}
					value={draft}
					onChange={(event) => setDraft(event.target.value)}
					placeholder={PLACEHOLDERS[surface]}
					onKeyDown={(event) => {
						if (event.key !== "Enter" || event.nativeEvent.isComposing) return;
						event.preventDefault();
						void onSend(surface, draft);
					}}
				/>
				<button
					type="submit"
					title="Send"
					disabled={busy || draft.trim().length === 0}
					className={draft.trim().length > 0 ? "is-live" : ""}
				>
					<ArrowUpIcon size={16} />
				</button>
			</form>
		</div>
	);
}

function SettingsSection({ title, children }: { readonly title: string; readonly children: ReactElement | ReactElement[] }): ReactElement {
	return <section className="settings-section"><h3>{title}</h3><div className="settings-section__card">{children}</div></section>;
}

function SettingRow({ label, description, children }: { readonly label: string; readonly description: string; readonly children: ReactElement }): ReactElement {
	return (
		<div className="settings-row">
			<div>
				<span className="settings-row__label">{label}</span>
				<span className="settings-row__desc">{description}</span>
			</div>
			<div className="settings-row__control">{children}</div>
		</div>
	);
}

function Switch({ checked, onChange }: { readonly checked: boolean; readonly onChange: () => void }): ReactElement {
	return <button type="button" role="switch" aria-checked={checked} className={`settings-switch${checked ? " is-on" : ""}`} onClick={onChange}><span /></button>;
}
