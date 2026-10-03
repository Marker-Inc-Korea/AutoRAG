# AutoRAG Lite MCP Server

AutoRAG Lite는 모델 없이 문서를 검색하고 인덱스를 관리하는 MCP 서버를 제공합니다.
MCP client는 `autorag-mcp`를 stdio 서버로 실행한 뒤 아래 tool을 호출합니다.

## 설치 및 실행

패키지를 설치합니다.

```bash
bun install -g @autorag/librarian
```

기존 Lite config를 지정해 서버를 실행합니다.

```bash
AUTORAG_CONFIG=/absolute/path/to/.autorag/config.json \
  autorag-mcp
```

Claude Desktop, Cursor, VS Code 등 stdio MCP 설정 예시:

```json
{
  "mcpServers": {
    "autorag": {
      "command": "autorag-mcp",
      "env": {
        "AUTORAG_CONFIG": "/absolute/path/to/.autorag/config.json"
      }
    }
  }
}
```

개발 checkout에서 직접 실행할 때:

```json
{
  "mcpServers": {
    "autorag": {
      "command": "bun",
      "args": ["/absolute/path/to/AutoRAG/src/mcp/index.ts"],
      "env": {
        "AUTORAG_CONFIG": "/absolute/path/to/project/.autorag/config.json"
      }
    }
  }
}
```

STDIO 서버의 stdout은 MCP protocol 전용입니다. 디버그 로그는 stderr로 출력해야 합니다.

## Tool 목록

| Tool | 설명 | 기본 상태 |
|---|---|---|
| `autorag.status` | 인덱스 상태와 freshness 조회 | 활성 |
| `autorag.search` | 설정된 corpus 검색 | 활성 |
| `autorag.refresh` | 인덱스 incremental refresh | 활성 |
| `autorag.report` | curated report 저장 | 활성 |
| `autorag.evidence` | report session의 저장된 근거 조회 | 활성 |
| `autorag.feedback` | 결과 번호별 useful/not-useful 기록 | 활성 |
| `autorag.duplicates` | 중복 문서 family read-only 스캔 | 활성 |

## 일반 workflow

1. `autorag.status`로 인덱스 상태 확인
2. 인덱스가 준비되지 않았으면 `autorag.refresh` 호출
3. `autorag.search` 호출
4. 결과의 `stale`, `diagnostics`, `unsearched`를 확인
5. 결과를 인용해 답변 작성
6. 장기적으로 보존할 답변이면 `autorag.report` 호출
7. 필요하면 `autorag.evidence`로 근거 확인
8. 사용자 평가를 받으면 `autorag.feedback` 호출

`autorag.search`는 기본적으로 자동 refresh하지 않습니다. stale index 결과도 반환하지만 `stale: true`와 diagnostics를 포함합니다. 최신 결과가 필수인 경우 먼저 `autorag.refresh`를 호출하거나 `strict: true`로 검색합니다.

## 검색 예시

```json
{
  "name": "autorag.search",
  "arguments": {
    "query": "refund exception approval policy",
    "topK": 5,
    "strict": false
  }
}
```

결과에는 다음 필드가 포함됩니다.

- `results`: 번호, source, retrieval method, score, metadata, content
- `stale`: 마지막 refresh가 현재 source 변경사항을 포함하는지 여부
- `unsearched`: 실행되지 않은 retrieval surface와 원래 오류
- `diagnostics`: degraded retrieval, stale index, component failure 정보

`unsearched` 또는 error diagnostic이 있으면 결과가 corpus 전체를 대표한다고 가정하지 마십시오.

## Report·evidence·feedback

`autorag.report`의 `mapping[].source`, `mapping[].method`, `mapping[].content`는 검색 결과에서 받은 값을 그대로 전달해야 합니다.
서버는 source 값을 다시 filesystem path로 해석하지 않고 memory에 저장합니다.

```json
{
  "name": "autorag.report",
  "arguments": {
    "query": "refund exception approval policy",
    "report": {
      "answer": "[1] Director approval is required before payout.",
      "results": [
        {
          "number": 1,
          "title": "Refund exception approval",
          "summary": "Director approval is required before payout.",
          "evidence": [
            { "excerpt": "Refund exceptions require director approval before payout." }
          ],
          "confidence": 0.9
        }
      ],
      "mapping": [
        {
          "number": 1,
          "source": "/workspace/docs/refund-policy.md",
          "method": "minsync",
          "content": "Refund exceptions require director approval before payout."
        }
      ]
    }
  }
}
```

성공하면 `sessionId`가 반환됩니다.

```json
{
  "name": "autorag.evidence",
  "arguments": {
    "sessionId": "<sessionId>",
    "resultNumber": 1
  }
}
```

```json
{
  "name": "autorag.feedback",
  "arguments": {
    "sessionId": "<sessionId>",
    "usefulNumbers": [1],
    "notUsefulNumbers": []
  }
}
```

feedback는 persisted memory를 사용하므로 MCP process를 재시작한 뒤에도 동작합니다.

## Read-only mode와 tool allowlist

쓰기 tool을 등록하지 않으려면:

```bash
AUTORAG_MCP_READ_ONLY=true autorag-mcp
```

read-only mode에서는 다음 tool만 노출됩니다.

```text
autorag.status
autorag.search
autorag.evidence
autorag.duplicates
```

개별 tool allowlist:

```bash
AUTORAG_MCP_TOOLS=autorag.status,autorag.search,autorag.evidence autorag-mcp
```

`AUTORAG_MCP_TOOLS`를 지정하면 목록에 있는 tool만 `tools/list`에 표시됩니다.

## 쓰기 범위와 보안

- 원본 source document는 수정·이동·삭제하지 않습니다.
- refresh가 쓰는 위치는 기존 AutoRAG Lite 계약에 따른 index/cache/memory 영역입니다.
- config는 `AUTORAG_CONFIG` 또는 기존 AutoRAG config 탐색 규칙으로 결정됩니다.
- MCP tool argument로 arbitrary workspace나 arbitrary file path를 받지 않습니다.
- datasource의 접근 범위는 config의 default-deny 정책을 따릅니다.
- API key와 credential은 tool 결과나 로그에 포함하지 않습니다.

현재 entrypoint는 local stdio용입니다. 원격 HTTP deployment에는 인증, Origin 검증, HTTPS, workspace별 authorization, rate limiting을 추가한 별도 transport 구성이 필요합니다.

## 오류 형식

입력 schema 오류는 MCP tool validation error로 반환됩니다. 실행 중 복구 가능한 오류는 `isError: true`와 구조화된 `errorCode`로 반환됩니다.

예:

```json
{
  "isError": true,
  "structuredContent": {
    "ok": false,
    "errorCode": "index-not-ready",
    "action": "autorag.refresh",
    "query": "refund policy"
  }
}
```

protocol 오류와 tool 실행 오류를 구분해 처리하십시오. `index-not-ready`, `stale-index`, `refresh-failed`, `session-not-found`는 모델이 다음 tool call을 결정할 수 있도록 반환됩니다.

## 검증

```bash
bun run typecheck
bun run build
bunx vitest run test/mcp/server.test.ts test/mcp/stdio.test.ts
```

`test/mcp/stdio.test.ts`는 실제 stdio MCP client로 refresh, search, report, evidence, process restart 후 feedback을 확인합니다.
