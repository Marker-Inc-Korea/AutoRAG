import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../styles/tokens.css";
import "../styles/base.css";
import "../styles/shell.css";
import "../styles/sidebar.css";
import "../styles/finder.css";
import "../styles/primitives.css";
import { App } from "./App";

const container = document.getElementById("root");
if (container === null) {
	throw new Error("Renderer mount point #root not found");
}

createRoot(container).render(
	<StrictMode>
		<App />
	</StrictMode>,
);
