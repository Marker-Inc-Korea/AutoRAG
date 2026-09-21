# KakaoTalk / lazykatok manual QA

This checklist validates the KakaoTalk datasource skill and its default-deny
tag access controls. Lazykatok does not expose per-source scope filtering.

## Preconditions

- `lazykatok` is installed and available on `PATH`, or a configured `LazykatokClient({ binaryPath })` points to it.
- KakaoTalk access is already granted to `lazykatok` outside AutoRAG.
- AutoRAG is configured with a `LazykatokSkill` and explicit trusted access:

```ts
new AutoRAGAgent({
  searchPaths: ["/docs"],
  datasourceSkills: [new LazykatokSkill({ instanceId: "personal" })],
  datasourceAccess: {
    allowedTags: ["kakaotalk"],
  },
});
```

## Required checks

1. **Default deny**
   - Remove `datasourceAccess`.
   - Run `agent.searchSingleDatasourceDocuments("kakao", "hello")`.
   - Expected: zero datasource results; no error, and no `search_datasource_kakao` tool is generated at all.

2. **Trusted allow**
   - Restore `allowedTags: ["kakaotalk"]`.
   - Run `agent.refresh()`.
   - Expected: `components.datasources` is `configured` or `degraded`.

3. **Datasource-native filtering**
   - Search with the trusted Kakao datasource.
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
