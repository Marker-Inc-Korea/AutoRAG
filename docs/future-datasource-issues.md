# Future datasource skill issues

Use `.github/ISSUE_TEMPLATE/datasource-skill.md` and labels `datasource-skill`, `integration`, and `데이터 소스 추가` for each integration. Add `manual-qa-required` when real credentials or local app permissions are needed.

## Proposed issues

1. Slack datasource skill — workspace/channel hierarchy, bot/user token scopes, message/thread indexing, file attachments.
2. Google Drive datasource skill — account/folder hierarchy, Docs/Sheets/PDF export parsing, shared-drive permissions.
3. Notion datasource skill — workspace/database/page hierarchy, block tree indexing, integration-token boundaries.
4. GitHub Issues/PRs datasource skill — owner/repo hierarchy, issue/PR/comment/review indexing, GitHub App scopes.
5. Gmail/IMAP datasource skill — account/label/folder hierarchy, email/thread/attachment indexing, OAuth or IMAP auth.
6. Discord datasource skill — guild/channel/thread hierarchy, bot permissions, message history windows.
7. Linear datasource skill — workspace/team/project hierarchy, issue/comment/document indexing.
8. Jira datasource skill — site/project/board hierarchy, issue/comment/attachment indexing.
9. Confluence datasource skill — site/space/page hierarchy, page tree and attachment indexing.
10. SharePoint/OneDrive datasource skill — tenant/site/library/folder hierarchy, Microsoft Graph scopes.
11. Dropbox datasource skill — account/team/folder hierarchy, file and Paper document indexing.
12. Local mail export datasource skill — mailbox/folder hierarchy for mbox/eml archives.
13. Browser bookmarks/history datasource skill — profile/folder/time-window hierarchy, local privacy controls.
14. Calendar datasource skill — account/calendar hierarchy, event/attendee/attachment indexing.
15. Obsidian vault datasource skill — vault/folder/tag hierarchy, markdown links and embeds.
16. Zotero datasource skill — library/collection hierarchy, item notes, metadata, and attachments.
17. RSS/news datasource skill — feed/category hierarchy, polling metadata and dedupe windows.

## Standard acceptance criteria

- Access is default-deny and server-bound.
- Tool arguments cannot grant `allowedTags` or `allowedScopes`.
- Results are filtered before merge and use slash-hierarchical source IDs.
- Diagnostics are non-throwing for expected auth/API failures.
- README/docs/manual QA are updated for the datasource.
