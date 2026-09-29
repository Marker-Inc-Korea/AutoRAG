import { useEffect, useState, type ReactElement } from "react";
import type { AccessRequest } from "../../../shared/settings-contract";
import type { SettingsBridge } from "../../../shared/settings-contract";
import { CloseIcon } from "./icons";
import { FileTile } from "./primitives/FileTile";

const settings = (window as unknown as { readonly autorag: { readonly settings: SettingsBridge } }).autorag.settings;
const fs = (window as unknown as { readonly autorag: { readonly fs: { reveal(path: string): Promise<void> } } }).autorag.fs;

export function RequestsPopover({ onClose, onChanged }: { readonly onClose: () => void; readonly onChanged?: () => void }): ReactElement {
	const [tab, setTab] = useState<"pending" | "done">("pending");
	const [requests, setRequests] = useState<readonly AccessRequest[]>([]);

	useEffect(() => {
		void settings.requestsList(tab).then(setRequests);
	}, [tab]);

	async function respond(id: string, allow: boolean): Promise<void> {
		await settings.requestsRespond(id, allow);
		setRequests((current) => current.filter((request) => request.id !== id));
		onChanged?.();
	}

	return (
		<>
			<button className="requests-popover__scrim" type="button" aria-label="Close Requests" onClick={onClose} />
			<section className="requests-popover" role="dialog" aria-modal="true" aria-label="Requests">
				<header className="requests-popover__header">
					<strong>Requests</strong>
					<nav className="requests-popover__tabs" aria-label="Request status">
						<button type="button" className={tab === "pending" ? "is-active" : ""} onClick={() => setTab("pending")}>대기 중 {tab === "pending" ? requests.length : ""}</button>
						<button type="button" className={tab === "done" ? "is-active" : ""} onClick={() => setTab("done")}>처리됨</button>
					</nav>
					<button type="button" className="icon-button" aria-label="Close Requests" title="Close" onClick={onClose}><CloseIcon /></button>
				</header>
				<div className="requests-popover__body">
					{requests.length === 0 ? (
						<div className="requests-popover__empty">
							<strong>{tab === "pending" ? "대기 중인 요청이 없습니다" : "처리된 요청이 없습니다"}</strong>
							<span>{tab === "pending" ? "새로운 접근 요청이 여기에 표시됩니다." : "처리된 요청이 여기에 표시됩니다."}</span>
						</div>
					) : requests.map((request) => (
						<article className="request-card" key={request.id}>
							<div className="request-card__topline">
								<div className="request-card__avatar">{request.contactName.slice(0, 1).toUpperCase()}</div>
								<div className="request-card__identity">
									<strong>{request.contactName}</strong>
									<span>{request.contactRole} · {request.requestedAt}</span>
								</div>
							</div>
							<p>{request.question}</p>
							<div className="request-card__files">
								{request.files.map((file) => (
									<div className="request-card__file" key={file.path}>
										<FileTile kind="pdf" size={16} />
										<div><strong>{file.name}</strong><small>{file.path}</small></div>
									</div>
								))}
							</div>
							<button type="button" className="request-card__reveal" onClick={() => void Promise.all(request.files.map((file) => fs.reveal(file.path)))}>Show in Finder</button>
							{request.status === "pending" ? (
								<div className="request-card__actions">
									<button type="button" className="request-card__deny" onClick={() => void respond(request.id, false)}>거부</button>
									<button type="button" className="request-card__allow" onClick={() => void respond(request.id, true)}>허용</button>
								</div>
							) : <span className={`request-card__status request-card__status--${request.status}`}>{request.status === "allowed" ? "Allowed" : "Denied"}</span>}
						</article>
					))}
				</div>
			</section>
		</>
	);
}
