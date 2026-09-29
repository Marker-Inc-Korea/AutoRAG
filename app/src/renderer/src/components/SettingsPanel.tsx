import { useEffect, useState, type FormEvent, type ReactElement } from "react";
import type { AppSettings, AssistantEvent, Contact, DataSourceRow } from "../../../shared/settings-contract";
import type { SettingsBridge } from "../../../shared/settings-contract";
import { CloseIcon } from "./icons";

const autorag = (window as unknown as { readonly autorag: { readonly settings: SettingsBridge } }).autorag;

type SettingsTab = "general" | "models" | "sources" | "contacts";

export function SettingsPanel({
	initialTab = "general",
	onClose,
}: {
	readonly initialTab?: SettingsTab;
	readonly onClose: () => void;
}): ReactElement {
	const [tab, setTab] = useState<SettingsTab>(initialTab);
	const [settings, setSettings] = useState<AppSettings | null>(null);
	const [contacts, setContacts] = useState<readonly Contact[]>([]);
	const [sources, setSources] = useState<readonly DataSourceRow[]>([]);
	const [contactDraft, setContactDraft] = useState({ id: "", name: "", role: "", description: "" });
	const [assistantText, setAssistantText] = useState("");
	const [assistantAnswer, setAssistantAnswer] = useState("");
	const [busy, setBusy] = useState(false);
	const [updateState, setUpdateState] = useState<"idle" | "checking" | "done">("idle");

	useEffect(() => {
		void Promise.all([
			autorag.settings.get().then(setSettings),
			autorag.settings.contactsList().then(setContacts),
			autorag.settings.sourcesList().then(setSources),
		]);
		const unsubscribe = autorag.settings.onEvent?.((event: AssistantEvent) => {
			if (event.type === "text") setAssistantAnswer((answer) => answer + event.delta);
			if (event.type === "done" || event.type === "error") setBusy(false);
			if (event.type === "error") setAssistantAnswer(event.message);
		});
		return unsubscribe;
	}, []);

	async function sendAssistant(event: FormEvent<HTMLFormElement>): Promise<void> {
		event.preventDefault();
		const text = assistantText.trim();
		if (text.length === 0 || busy) return;
		setBusy(true);
		setAssistantAnswer("");
		setAssistantText("");
		await autorag.settings.assistantSend(tab === "sources" ? "sources" : "models", text);
	}

	async function toggleSetting(key: keyof AppSettings): Promise<void> {
		if (settings === null || typeof settings[key] !== "boolean") return;
		const next = await autorag.settings.set({ [key]: !settings[key] });
		setSettings(next);
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

	return (
		<div className="settings-backdrop" role="dialog" aria-modal="true" aria-label="Settings">
			<section className="settings-panel">
				<header className="settings-panel__header">
					<strong>Settings</strong>
					<nav className="settings-panel__tabs" aria-label="Settings sections">
						{(["general", "models", "sources"] as const).map((item) => (
							<button type="button" className={tab === item ? "is-active" : ""} key={item} onClick={() => setTab(item)}>
								{item === "sources" ? "Data Sources" : item[0]?.toUpperCase() + item.slice(1)}
							</button>
						))}
					</nav>
					<button type="button" className="icon-button" aria-label="Close Settings" title="Close" onClick={onClose}><CloseIcon /></button>
				</header>
				<div className="settings-panel__body">
					{tab === "general" && settings !== null ? (
						<>
							<SettingsSection title="General">
								<SettingRow label="Language" description="앱 인터페이스 언어">
									<div className="settings-select"><select value={settings.language} onChange={(event) => void autorag.settings.set({ language: event.target.value as AppSettings["language"] }).then(setSettings)}>
										<option value="ko">한국어</option><option value="en">English</option><option value="ja">日本語</option><option value="zh">简体中文</option>
									</select></div>
								</SettingRow>
								<SettingRow label="Theme" description="앱 테마">
									<div className="settings-select"><select value={settings.theme} onChange={(event) => void autorag.settings.set({ theme: event.target.value as AppSettings["theme"] }).then(setSettings)}>
										<option value="light">Light</option><option value="dark">Dark</option><option value="system">System</option>
									</select></div>
								</SettingRow>
								<SettingRow label="Global shortcut" description="어디서든 AutoRAG Agent 열기">
									<div className="settings-keycaps"><kbd>⌥</kbd><kbd>Space</kbd></div>
								</SettingRow>
								<SettingRow label="Launch at login" description="Mac 로그인 시 자동 실행">
									<Switch checked={settings.launchAtLogin} onChange={() => void toggleSetting("launchAtLogin")} />
								</SettingRow>
								<SettingRow label="Show in menu bar" description="메뉴 막대에 AutoRAG 표시">
									<Switch checked={settings.showInMenuBar} onChange={() => void toggleSetting("showInMenuBar")} />
								</SettingRow>
							</SettingsSection>
							<SettingsSection title="Software Updates">
								<SettingRow label="AutoRAG Agent 1.4.2" description={updateState === "done" ? "최신 버전입니다 · 방금 확인" : "업데이트를 자동으로 확인합니다"}>
									<button type="button" onClick={() => { setUpdateState("checking"); window.setTimeout(() => setUpdateState("done"), 500); }}>{updateState === "checking" ? "Checking…" : "Check for Updates"}</button>
								</SettingRow>
								<SettingRow label="Automatically install updates" description="백그라운드에서 다운로드 후 재시작 시 적용">
									<Switch checked={settings.autoInstallUpdates} onChange={() => void toggleSetting("autoInstallUpdates")} />
								</SettingRow>
								<SettingRow label="Beta updates" description="새 기능을 먼저 받아보기">
									<Switch checked={settings.betaUpdates} onChange={() => void toggleSetting("betaUpdates")} />
								</SettingRow>
							</SettingsSection>
							<SettingsSection title="Privacy">
								<SettingRow label="Clear chat history" description="저장된 모든 AI Search 대화를 삭제">
									<button type="button" className="settings-danger-button" onClick={() => void autorag.settings.clearHistory()}>Clear…</button>
								</SettingRow>
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
						<>
							<div className="settings-intro"><strong>Connected sources</strong><span>{sources.length} {sources.length === 1 ? "source" : "sources"} configured from AutoRAG</span></div>
							{sources.length === 0 ? <p>No configured data sources.</p> : sources.map((source) => (
								<label className="settings-card settings-source-card" key={source.id}>
									<Switch checked={source.enabled} onChange={() => void toggleSource(source)} />
									<div className="settings-source-card__body"><strong>{source.name}</strong><span>{source.detail} · {source.status}</span><small>{source.description}</small>{source.progress === null ? null : <progress value={source.progress} max={100} />}</div>
									<span className="settings-source-card__chevron">›</span>
								</label>
							))}
							<AssistantBox tab={tab} assistantAnswer={assistantAnswer} assistantText={assistantText} busy={busy} setAssistantText={setAssistantText} onSubmit={sendAssistant} />
						</>
					) : null}
					{tab === "models" ? (
						<>
							<div className="settings-model-card"><span className="settings-model-tile">A</span><div><strong>AutoRAG Agent</strong><span>Connected model · Quick + Deep shared</span></div><em>Quick · Deep</em></div>
							<AssistantBox tab={tab} assistantAnswer={assistantAnswer} assistantText={assistantText} busy={busy} setAssistantText={setAssistantText} onSubmit={sendAssistant} />
						</>
					) : null}
				</div>
			</section>
		</div>
	);
}

function SettingsSection({ title, children }: { readonly title: string; readonly children: ReactElement | ReactElement[] }): ReactElement {
	return <section className="settings-section"><h3>{title}</h3><div className="settings-section__card">{children}</div></section>;
}

function SettingRow({ label, description, children }: { readonly label: string; readonly description: string; readonly children: ReactElement }): ReactElement {
	return <div className="settings-row"><div><strong>{label}</strong><span>{description}</span></div><div className="settings-row__control">{children}</div></div>;
}

function Switch({ checked, onChange }: { readonly checked: boolean; readonly onChange: () => void }): ReactElement {
	return <button type="button" role="switch" aria-checked={checked} className={`settings-switch${checked ? " is-on" : ""}`} onClick={onChange}><span /></button>;
}

function AssistantBox({
	tab,
	assistantAnswer,
	assistantText,
	busy,
	setAssistantText,
	onSubmit,
}: {
	readonly tab: "models" | "sources";
	readonly assistantAnswer: string;
	readonly assistantText: string;
	readonly busy: boolean;
	readonly setAssistantText: (text: string) => void;
	readonly onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}): ReactElement {
	return (
		<>
			<div className="settings-assistant__answer">{assistantAnswer || "Ask the AutoRAG settings assistant about this configuration."}</div>
			<form className="settings-assistant" onSubmit={onSubmit}>
				<input value={assistantText} onChange={(event) => setAssistantText(event.target.value)} placeholder={`Ask about ${tab}...`} />
				<button type="submit" disabled={busy}>Send</button>
			</form>
		</>
	);
}
