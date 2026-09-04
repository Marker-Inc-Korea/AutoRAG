# Future datasource skill issues

Use `.github/ISSUE_TEMPLATE/datasource-skill.md` and labels `datasource-skill`, `integration`, and `데이터 소스 추가` for each integration. Add `manual-qa-required` when real credentials or local app permissions are needed.

## Proposed issues

1. Slack datasource skill — workspace/channel hierarchy, bot/user token scopes, message/thread indexing, file attachments.
2. Cloud-drive datasource skill — provider-neutral account/folder hierarchy, file parsing, and shared-drive permissions.
3. Notion datasource skill — workspace/database/page hierarchy, block tree indexing, integration-token boundaries.
4. GitHub Issues/PRs datasource skill — owner/repo hierarchy, issue/PR/comment/review indexing, GitHub App scopes.
5. Linear datasource skill — workspace/team/project hierarchy, issue/comment/document indexing.
6. Jira datasource skill — site/project/board hierarchy, issue/comment/attachment indexing.
7. Confluence datasource skill — site/space/page hierarchy, page tree and attachment indexing.
8. SharePoint/OneDrive datasource skill — tenant/site/library/folder hierarchy, Microsoft Graph scopes.
9. Dropbox datasource skill — account/team/folder hierarchy, file and Paper document indexing.
10. Local mail export datasource skill — mailbox/folder hierarchy for mbox/eml archives.
11. Browser bookmarks/history datasource skill — profile/folder/time-window hierarchy, local privacy controls.
12. Calendar datasource skill — account/calendar hierarchy, event/attendee/attachment indexing.
13. Obsidian vault datasource skill — vault/folder/tag hierarchy, markdown links and embeds.
14. Zotero datasource skill — library/collection hierarchy, item notes, metadata, and attachments.
15. RSS/news datasource skill — feed/category hierarchy, polling metadata and dedupe windows.

## Standard acceptance criteria

- Access is default-deny and server-bound.
- Tool arguments cannot grant `allowedTags` or `allowedScopes`.
- Results are filtered before merge and use slash-hierarchical source IDs.
- Diagnostics are non-throwing for expected auth/API failures.
- README/docs/manual QA are updated for the datasource.
