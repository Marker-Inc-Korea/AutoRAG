# Future datasource skill issues

Use `.github/ISSUE_TEMPLATE/datasource-skill.md` and labels `datasource-skill`, `integration`, and `데이터 소스 추가` for each integration. Add `manual-qa-required` when real credentials or local app permissions are needed.

## Proposed issues

Lark/Feishu v1 is not in this list: it ships as remote search through `lark-cli` (issue #1672). A later local archive is a separate issue. Local materialization is intentionally out of scope for v1.

1. Slack datasource skill — workspace/channel hierarchy, bot/user token scopes, message/thread indexing, file attachments.
2. Notion datasource skill — workspace/database/page hierarchy, block tree indexing, integration-token boundaries.
3. GitHub Issues/PRs datasource skill — owner/repo hierarchy, issue/PR/comment/review indexing, GitHub App scopes.
4. Linear datasource skill — workspace/team/project hierarchy, issue/comment/document indexing.
5. Jira datasource skill — site/project/board hierarchy, issue/comment/attachment indexing.
6. Confluence datasource skill — site/space/page hierarchy, page tree and attachment indexing.
7. SharePoint/OneDrive datasource skill — tenant/site/library/folder hierarchy, Microsoft Graph scopes.
8. Dropbox datasource skill — account/team/folder hierarchy, file and Paper document indexing.
9. Local mail export datasource skill — mailbox/folder hierarchy for mbox/eml archives.
10. Browser bookmarks/history datasource skill — profile/folder/time-window hierarchy, local privacy controls.
11. Calendar datasource skill — account/calendar hierarchy, event/attendee/attachment indexing.
12. Obsidian vault datasource skill — vault/folder/tag hierarchy, markdown links and embeds.
13. Zotero datasource skill — library/collection hierarchy, item notes, metadata, and attachments.
14. RSS/news datasource skill — feed/category hierarchy, polling metadata and dedupe windows.

## Standard acceptance criteria

- Connections are server-bound and every configured connection is searchable; there is no access layer.
- Tool arguments cannot change trusted connector configuration; a query `scope` can only narrow, never widen.
- Results are narrowed by query scope before merge and use slash-hierarchical source IDs.
- Diagnostics are non-throwing for expected auth/API failures.
- README/docs/manual QA are updated for the datasource.
