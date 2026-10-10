# KakaoTalk / lazykatok manual QA

This checklist validates the KakaoTalk datasource skill and its
connected-datasource behavior. Lazykatok does not expose per-source scope
filtering.

## Preconditions

- `lazykatok` is installed and available on `PATH`, or a configured `LazykatokClient({ binaryPath })` points to it.
- KakaoTalk access is already granted to `lazykatok` outside AutoRAG.
- AutoRAG is configured with a `LazykatokSkill`:

```ts
new AutoRAGAgent({
  searchPaths: ["/docs"],
  datasourceSkills: [new LazykatokSkill({ instanceId: "personal" })],
});
```

## Required checks

1. **Configured connection is searchable**
   - Run `agent.searchSingleDatasourceDocuments("kakao", "hello")`.
   - Expected: the generated `search_datasource_kakao` tool is registered and the search returns lazykatok results, or a diagnostic when the archive is empty; no error is thrown.

2. **Refresh status**
   - Run `agent.refresh()`.
   - Expected: `components.datasources` is `configured` or `degraded`.

3. **Datasource-native filtering**
   - Search with the configured Kakao datasource.
   - Expected: lazykatok-owned chat/channel filtering and chat identity metadata
     remain intact; AutoRAG does not apply a virtual source scope.

4. **Missing binary / permission failure**
   - Point `LazykatokClient` at a nonexistent binary or run without required OS permissions.
   - Expected: no throw; the failure surfaces as a warning/error diagnostic.

5. **Public response curation**
   - Run `searchDocuments()` and let the librarian curate a KakaoTalk-supported answer.
   - Expected: visible `answer` and `results` contain curated facts grounded in the datasource evidence.

## Environment limitation note

CI and most development containers do not have a real KakaoTalk profile or macOS app-container permissions. In those environments, perform checks 1 and 4 with the test/fake lazykatok client; record real-data checks as manually blocked by missing local KakaoTalk credentials rather than bypassing the safety requirements.
