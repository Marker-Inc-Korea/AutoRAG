import type { ReactElement } from "react";
import { APP_NAME, formatWindowTitle } from "../../shared/app-info";

export function App(): ReactElement {
	return (
		<div className="app-shell">
			<h1>{formatWindowTitle(APP_NAME, window.autorag.version)}</h1>
		</div>
	);
}
