# KakaoTalk / katok manual QA

This checklist validates the KakaoTalk datasource without exposing local KakaoTalk storage paths or personal identifiers.

## Preconditions

- `katok` is installed and available on `PATH`, or a configured `KatokClient({ binaryPath })` points to it.
- KakaoTalk access is already granted to `katok` outside AutoRAG.
- AutoRAG is configured with a `KatokSkill` and explicit trusted access:

```ts
new AutoRAGAgent({
  searchPaths: ["/docs"],
  datasourceSkills: [new KatokSkill({ instanceId: "personal" })],
  datasourceAccess: {
    allowedTags: ["kakaotalk"],
    allowedScopes: ["/kakao/personal/**"],
  },
});
```

## Required checks

1. **Default deny**
   - Remove `datasourceAccess`.
   - Run `searchDatasourceDocuments("hello")`.
   - Expected: zero datasource results; no error; no path/PII in output.

2. **Trusted allow**
   - Restore `allowedTags: ["kakaotalk"]` and an instance scope such as `/kakao/personal/**`.
   - Run `agent.refresh()`.
   - Expected: `components.datasources` is `configured` or `degraded`; any diagnostic is path-opaque.

3. **Scope narrowing**
   - Search with a narrower scope inside the trusted instance.
   - Expected: only matching opaque `/kakao/personal/...` results survive.
   - Search with a scope outside trusted access, e.g. `/kakao/other/**`.
   - Expected: zero results.

4. **Remote embedding egress rejection**
   - Set one of these before refresh/search: `EMBEDDER_BASE_URL`, `embedder_base_url`, `ALLOW_REMOTE_EMBEDDINGS`, `allow_remote_embeddings`, or URL-valued `KATOK_EMBEDDER`.
   - Expected: `katok` is not spawned; datasource diagnostic code maps to egress rejection; the remote URL value is not present in any serialized result/status.

5. **Missing binary / permission failure**
   - Point `KatokClient` at a nonexistent binary or run without required OS permissions.
   - Expected: no throw; diagnostic is warning/error and does not contain `/Users`, `Library/Containers`, `com.kakao`, account IDs, or phone numbers.

6. **Public response opacity**
   - Run `searchDocuments()` and let the librarian curate a KakaoTalk-supported answer.
   - Expected: visible `answer` and `results` contain curated facts only. Internal opaque sources may appear only in hidden mapping/session state, never in visible text.

## Environment limitation note

CI and most development containers do not have a real KakaoTalk profile or macOS app-container permissions. In those environments, perform checks 1, 4, 5, and path-opacity assertions with the test/fake katok client; record real-data checks as manually blocked by missing local KakaoTalk credentials rather than bypassing the safety requirements.
