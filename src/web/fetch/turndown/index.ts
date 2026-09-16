/**
 * Vendored from oh-my-pi (packages/utils/src/turndown) — MIT licensed.
 * Behavior-compatible Turndown reimplementation used as the `native`
 * HTML-to-markdown backend by `src/web/fetch/html-renderer.ts`.
 *
 * Source: https://github.com/oh-my-pi/oh-my-pi
 */

export * from "./gfm.ts";
export { default, default as TurndownService } from "./service.ts";
export * from "./types.ts";
