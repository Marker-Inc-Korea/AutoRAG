# AutoRAG for always-on cloud agents (Grok Bot, OpenAI dots, Claude Duet, Meta Muse)

Hermes, Claude Code and Codex start `autorag-mcp` as a local child process (stdio). Always-on personal agents
(Grok Bot, OpenAI dots, Claude Duet, Meta Muse) run in their vendor's cloud and connect to an MCP server **by URL**.
`autorag-mcp --http` is that URL: the same tools as stdio, served over streamable HTTP, with your index staying on
your machine. The agent's model reads the chunks a search returns, so they go wherever that agent runs.

## Start the server

```bash
export AUTORAG_MCP_TOKEN="$(openssl rand -base64 32)"   # optional; see "Owner token"
autorag-mcp --http --port 8765                           # http://127.0.0.1:8765/mcp, read-only
```

Cloud agents cannot reach `127.0.0.1`. Put an HTTPS tunnel or reverse proxy in front and tell the server its public
origin. That turns on the OAuth endpoints connector-only clients need and widens the Host/Origin allowlist to that
host:

```bash
cloudflared tunnel --url http://127.0.0.1:8765           # prints https://<random>.trycloudflare.com
autorag-mcp --http --port 8765 --public-url https://<random>.trycloudflare.com
```

A quick tunnel's URL changes on every start; use a named tunnel (or Tailscale Funnel, ngrok with a reserved domain)
for a stable one. Index first (`autorag lite refresh`): the server serves what the index already holds.

| Flag / env | Effect |
| --- | --- |
| `--host`, `--port` | Bind address (default `127.0.0.1:8765`; `--port 0` picks a free port). A non-loopback `--host` without `--public-url` is refused. |
| `--public-url` / `AUTORAG_MCP_PUBLIC_URL` | HTTPS origin agents reach you at. Enables OAuth and the Host/Origin allowlist for it. |
| `--allow-write` | Also offer `autorag.refresh`, `autorag.report`, `autorag.convert`. Default is **read-only**: a remote agent gets search and evidence only. |
| `AUTORAG_MCP_TOOLS` | Comma-separated exact tool allowlist (same as stdio). |
| `AUTORAG_MCP_TOKEN` | Owner token, at least 32 characters. |

### Owner token

One secret opens everything. Without `AUTORAG_MCP_TOKEN`, the server creates a random one at
`$AUTORAG_HOME/mcp-http-token` (mode 0600) and prints only the path. It is:

- the `Authorization: Bearer <token>` credential for clients that can send a header (Claude Code, Codex, Hermes, and
  Grok Bot's custom-header option);
- what you type on the consent page when an agent connects through OAuth;
- the key every OAuth token is signed with. **Rotating it revokes every issued token.**

### What the server enforces

- Every `/mcp` request needs a valid bearer token (constant-time compare); `/healthz` answers `{"ok":true}` and nothing else.
- `Host` must be loopback or the public URL's host; a browser `Origin` must be the public origin (DNS-rebinding guard).
- OAuth (only with `--public-url`): dynamic client registration for public clients, authorization code + PKCE `S256`
  only, https (or loopback) redirect URIs only, the redirect must be one registered for that client, tokens are bound to
  the `/mcp` resource, codes are single use and live 60 s, access tokens 1 h, refresh tokens 30 d. The consent page
  throttles wrong owner tokens (5 per minute) and refuses framing.
- Request bodies over 1 MiB are refused.
- Tokens are stateless HMACs, so a restart keeps them valid; there is no per-token revocation other than rotating the owner token.

### Check the URL before you paste it into a vendor dialog

```bash
autorag-mcp --check https://<public>/mcp          # human-readable; exit 1 if any step fails
autorag-mcp --check https://<public>/mcp --json   # one JSON document
```

`--check` replays what a connector-only agent does before its first tool call: the 401 challenge, the protected-resource
and authorization-server metadata (and that they name the URL you gave), client registration, the consent page, and, if
`AUTORAG_MCP_TOKEN` (or the stored token file) is available, a real `tools/list`. It catches a tunnel pointing at the
wrong port or a `--public-url` that does not match the tunnel, which a vendor dialog reports only as "couldn't connect".
It never types the owner token into the consent page and never prints it. Write tools being exposed is a warning, not a
failure.

## Per agent

The vendor steps below come from each vendor's own documentation as of 2026-11-12 and were **not** executed against
the vendor's cloud (those need an account on each service). Compatibility is verified at the protocol level in
`test/mcp/http.test.ts` and over a real Cloudflare tunnel with real Claude Code, Codex and Hermes clients; see the PR.

### Claude Duet (and Claude / Cowork)

Duet inherits the connectors on your Claude account. Claude connects from Anthropic's cloud, so the URL must be
public. In **Customize > Connectors > Add custom connector**: URL `https://<public>/mcp`, authentication **Sign in
now**, OAuth client **Register automatically** (this server supports dynamic registration, not client-ID metadata
documents). Approve on the AutoRAG consent page with the owner token. Claude's optional "Request headers" beta can
send `Authorization: Bearer <owner token>` instead of OAuth.

Claude Code and Desktop's local config: `claude mcp add --transport http autorag https://<public>/mcp --header "Authorization: Bearer $AUTORAG_MCP_TOKEN"`.

### OpenAI dots (ChatGPT / Codex plugins)

A dot uses the plugins on your ChatGPT account. In ChatGPT **Plugins > + > Add custom MCP server**, enter
`https://<public>/mcp` and choose OAuth. ChatGPT cannot send a custom API key, so OAuth is required; this server
publishes the protected-resource and authorization-server metadata ChatGPT discovers, and registers it dynamically.
Then **Create as a plugin**; ask the dot to use AutoRAG. Codex on your computer can use the header form:
`codex mcp add autorag --url https://<public>/mcp --bearer-token-env-var AUTORAG_MCP_TOKEN`.

### Grok Bot

Copy-paste prompt: [`plugins/remote-agents/connect/grok-bot.md`](../plugins/remote-agents/connect/grok-bot.md).


Ask a Bot to add a custom MCP server (or use Plugins and paste the URL). Give it `https://<public>/mcp`, then either
sign in through OAuth or add the header `Authorization: Bearer <owner token>`. All of an account's Bots share one
cloud computer and the connected apps, so a connected AutoRAG is visible to every Bot on that account; connect it
read-only (the default) and keep it off accounts you share.

### Meta Muse

Copy-paste prompt: [`plugins/remote-agents/connect/muse.md`](../plugins/remote-agents/connect/muse.md).


Ask Muse in chat to create a custom connector for an MCP server at `https://<public>/mcp`. Muse keeps credentials in
its Secure Credentials Store. Prefer the OAuth sign-in (the owner token is typed only on your own consent page);
Meta does not review custom connectors, so keep the server read-only.

## Limits

- The server serves one corpus, yours. There are no per-user accounts: anyone holding the owner token (or a token
  issued by it) can search everything the config indexes. Do not hand a shared agent the token.
- A Duet/dots/Muse/Grok Bot request reaches you only while the server and tunnel are up. The agents run in the cloud,
  your index does not.
- Datasource tools (Slack, Kakao, mail, ...) are exposed like any other search tool. Leave datasources you would not
  show a cloud model out of the config you serve, or narrow with `AUTORAG_MCP_TOOLS`.
