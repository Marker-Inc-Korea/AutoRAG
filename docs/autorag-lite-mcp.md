# AutoRAG Lite MCP Server

AutoRAG Lite는 모델 없이 문서를 검색하고 인덱스를 관리하는 MCP 서버를 제공합니다.
MCP client는 `autorag-mcp`를 stdio 서버로 실행한 뒤 아래 tool을 호출합니다.
서버 구조를 한눈에 보려면 [MCP 계약 시각화 문서](autorag-lite-mcp-contract.html) 또는 [계약 상세 Markdown](autorag-lite-mcp-contract.md)을 확인하십시오.

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

| Tool | 설명 | read-only |
|---|---|---|
| `autorag.status` | 인덱스 상태와 freshness 조회 | 예 |
| `autorag.search` | 설정된 corpus를 선택적으로 검색 | 예 |
| `autorag.search.files` | OS에 맞는 파일/폴더 이름 검색: Windows는 bundled Everything, macOS/Linux는 configured-root filesystem walker (내용 미열람, literal substring) | 예 |
| `autorag.datasources.list` | 권한이 부여된 datasource catalog 조회 | 예 |
| `autorag.datasources.get` | datasource 하나의 catalog 항목 조회 | 예 |
| `autorag.duplicates` | 설정된 검색 root의 exact/near 중복 문서 family를 Dupey로 스캔 | 예 |
| `autorag.search_datasource_<id>` | 통합된 authorized datasource **하나만** 검색 (catalog별 동적 생성) | 예 |
| `autorag.refresh` | 인덱스 incremental refresh | 아니오 |

## 일반 workflow

1. `autorag.status`로 인덱스 상태 확인
2. 인덱스가 준비되지 않았으면 `autorag.refresh` 호출
3. `autorag.search`로 corpus 검색 (또는 파일 이름만 필요하면 `autorag.search.files`)
4. 결과의 `stale`, `diagnostics`, `unsearched`를 확인
5. 중복/개정본 정리가 필요하면 `autorag.duplicates`로 exact/near family를 확인
6. 결과를 인용해 답변 작성

`autorag.search`는 기본적으로 자동 refresh하지 않습니다. parsed mirror refresh가 아직 완료되지 않았으면 `index-not-ready` 오류를 돌려주므로 `autorag.refresh`를 먼저 호출합니다. stale index 결과도 반환하지만 `stale: true`와 diagnostics를 포함합니다. 최신 결과가 필수인 경우 먼저 `autorag.refresh`를 호출하거나 `strict: true`로 검색합니다.

## autorag.search — 선택 검색

```json
{
  "name": "autorag.search",
  "arguments": {
    "query": "refund exception approval policy",
    "topK": 5,
    "scope": "/kakao/default",
    "tags": ["kakao"],
    "strict": false,
    "datasourceIds": ["kakao"],
    "methods": ["kakao.keyword"],
    "local": false
  }
}
```

입력 필드:

- `query` (필수): 검색 문자열
- `topK`: 반환할 최대 결과 수
- `scope`: 이미 권한이 부여된 scope 안에서만 좁히는 선택적 범위
- `tags`: datasource 접근을 좁히는 태그
- `strict`: `true`이면 stale index 결과를 거부
- `datasourceIds`: 실행할 datasource id 목록
- `methods`: 실행할 retrieval method 이름 목록
- `local`: local (non-datasource) method 실행 여부

선택 semantics (권한을 넓힐 수 없음):

- `datasourceIds`를 생략하면 권한이 부여된 모든 datasource method가 대상입니다.
- `datasourceIds`를 지정하면 그 datasource의 method만 대상이며, `local === true`일 때만 local method가 추가됩니다.
- `local === false`이면 local method는 절대 실행되지 않습니다.
- `methods`를 지정하면 선택된 집합과 교집합되며, 각 method는 존재하고 (datasource method라면) 권한이 있어야 합니다.
- 알 수 없거나 권한이 없는 datasource/method를 지정하면 selection 오류로 거부됩니다.
- 선택은 backend 실행 **이전**에 적용되므로, 선택되지 않았거나 권한이 없는 datasource의 backend는 실행되지 않습니다.

결과에는 다음 필드가 포함됩니다.

- `results`: 번호, source, retrieval method, score, metadata, content
- `selection`: 실제 적용된 datasource/method/local 선택
- `stale`: 마지막 refresh가 현재 source 변경사항을 포함하는지 여부
- `unsearched`: 실행되지 않은 retrieval surface와 원래 오류
- `diagnostics`: degraded retrieval, stale index, component failure 정보

`unsearched` 또는 error diagnostic이 있으면 결과가 corpus 전체를 대표한다고 가정하지 마십시오.

## autorag.search.files — 파일/폴더 이름 검색 (OS 공통)

```json
{
  "name": "autorag.search.files",
  "arguments": {
    "query": "refund",
    "root": "docs",
    "matchPath": false,
    "matchCase": false,
    "kind": "files",
    "maxResults": 50,
    "offset": 0
  }
}
```

- 하나의 portable tool입니다. 서버가 실행 OS를 감지해 backend를 자동 선택합니다: Windows에서는 workspace별로 격리된 bundled voidtools Everything index를, macOS/Linux에서는 설정된 root를 직접 순회하는 filesystem walker를 사용합니다(외부 `fsearch` 같은 별도 binary가 아닙니다).
- `query`는 정규식이 아니라 literal substring으로 취급됩니다. Windows에서는 이 literal을 escape해 Everything regex(`regex: true`)로 변환하므로, 어느 OS에서도 동일하게 substring 일치합니다. wildcards/`ext:`/`dm:` 같은 Everything 전용 문법은 tool 입력으로 노출되지 않습니다.
- `root`는 config의 searchPath 또는 그 안의 실제 하위 디렉터리만 선택할 수 있습니다. realpath 기반 containment로 검증하므로 root 밖으로 범위를 넓힐 수 없고, `excludePaths`와 내부 디렉터리(symlink escape, `.git`, `.autorag` 등)는 결과에서 제외됩니다.
- `matchPath`가 `true`이면 파일 이름 대신 검색 root 기준 상대 경로에 대해 일치시킵니다.
- `kind`는 `"files"` 또는 `"folders"`로 결과를 좁힙니다.
- `maxResults`/`offset`은 권한 필터링 이후 페이지에 적용되며, lookahead 한 칸으로 `truncated`를 판정합니다.
- 이 tool은 **파일 내용을 읽지 않습니다.** 디렉터리 목록과 이름만 조회합니다.
- parsed mirror refresh가 없어도 동작합니다. 이름 검색은 index readiness에 의존하지 않습니다.
- 결과 형식은 `{ ok, backend, results, truncated, diagnostics }`입니다. `backend` discriminator는 `"filesystem"` 또는 `"everything"`이며, `results`의 각 항목은 `{ path, type }`입니다.
- Windows에서 Everything provider가 비활성/실패하면 filesystem walker로 **조용히 대체하지 않고** `{ ok: false, backend: "everything", reason, message }`를 `isError: true`로 반환합니다. platform이 Windows가 아니면(예: 테스트 override) filesystem backend가 선택됩니다.
- root가 없거나 범위를 벗어나면 빈 결과와 warning diagnostic을 반환하며, 임의의 전역 `/` 스캔으로 대체하지 않습니다.
- Everything은 관리자 권한이나 서비스 설치를 요구하지 않는 사용자 수준 instance입니다.

## Datasource 목록 조회

`autorag.datasources.list`와 `autorag.datasources.get`은 read-only이며 datasource를 실행하거나 corpus를 수정하지 않습니다.

```json
{ "name": "autorag.datasources.list", "arguments": {} }
```

```json
{ "name": "autorag.datasources.get", "arguments": { "datasourceId": "kakao" } }
```

각 항목은 identity, capability tags, 권한이 부여된 source scope 문자열만 담습니다. credential, config 경로, raw instance metadata는 포함되지 않습니다. 존재하지 않거나 권한이 없는 id는 `datasource-not-found` 오류로 반환됩니다.

## autorag.duplicates — 중복 문서 스캔

```json
{ "name": "autorag.duplicates", "arguments": {} }
```

- config의 `searchPaths` 각 root를 Dupey(`scan <root> --json`)로 스캔합니다. 입력 인자가 없습니다.
- 결과는 `roots`, `exactGroups`(`{ hash, files }` — canonical extracted text hash가 같은 파일 묶음), `families`(near/contains 포함), `extractionErrors`, 그리고 항상 `action: "review"`를 담습니다.
- **read-only입니다.** 원본 파일을 이동·삭제하지 않으며, exact group도 자동 정리 대상이 아닙니다. `families` 중 near/contains는 삭제 근거가 아니므로 반드시 검토하세요.
- Dupey 실행이 실패하면(미설치, timeout, 잘못된 출력) `duplicates-failed` 오류를 `retryable: true`로 반환합니다.
- Dupey 실행 설정은 config의 `dupey` 항목(`binaryPath`, `timeoutMs`, `enabled`)으로 지정할 수 있습니다.

## 동적 datasource 검색 tool

서버는 catalog에서 **권한이 부여되었고 retrieval method가 등록된** datasource마다 `autorag.search_datasource_<sanitized-id>` tool을 하나씩 생성합니다. `<sanitized-id>`는 datasource id를 lowercase로 바꾸고 영숫자 외 문자를 `_`로 접은 뒤 앞뒤 `_`를 제거한 값입니다 (예: `my.kakao` → `search_datasource_my_kakao`).

```json
{
  "name": "autorag.search_datasource_spotlight",
  "arguments": { "query": "refund", "topK": 5, "scope": "/spotlight/default" }
}
```

- 입력은 `query`(필수), `topK`, `scope`뿐입니다. datasource 선택과 authorization은 서버 config에서 결정되므로 인자로 넓힐 수 없습니다.
- 이 tool은 `searchSelected`를 `{ datasourceIds: [<id>], local: false }`로 호출합니다. 따라서 해당 datasource의 method만 실행되고 local method는 절대 실행되지 않습니다.
- tool이 없으면 그 datasource는 catalog에 없거나(비권한) retrieval method가 등록되지 않은 것입니다. `autorag.datasources.list`로 확인하세요.
- `index-not-ready`, `invalid-selection`, `search-failed`는 `autorag.search`와 동일한 규칙으로 반환되며, 오류에 `datasourceId`가 포함됩니다.

## Read-only mode와 tool allowlist

쓰기 tool을 등록하지 않으려면:

```bash
AUTORAG_MCP_READ_ONLY=true autorag-mcp
```

read-only mode에서는 `autorag.refresh`가 노출되지 않습니다. 나머지 검색·조회 tool은 모두 read-only입니다.

```text
autorag.status
autorag.search
autorag.search.files
autorag.datasources.list
autorag.datasources.get
autorag.duplicates
autorag.search_datasource_<id>
```

개별 tool allowlist:

```bash
AUTORAG_MCP_TOOLS=autorag.status,autorag.search,autorag.datasources.list autorag-mcp
```

`AUTORAG_MCP_TOOLS`를 지정하면 목록에 있는 tool만 `tools/list`에 표시됩니다.

## 쓰기 범위와 보안

- 원본 source document는 수정·이동·삭제하지 않습니다. `autorag.search.files`(Windows에서는 read-only Everything index, 그 외에는 filesystem walker)와 `autorag.duplicates`는 파일 내용을 읽지 않거나 이름/메타데이터만 read-only로 조회합니다.
- refresh가 쓰는 위치는 기존 AutoRAG Lite 계약에 따른 index/cache 영역입니다.
- config는 `AUTORAG_CONFIG` 또는 기존 AutoRAG config 탐색 규칙으로 결정됩니다.
- MCP tool argument로 arbitrary workspace나 arbitrary file path를 받지 않습니다. 파일 검색의 `root`는 설정된 root 안으로만 한정됩니다.
- datasource의 접근 범위는 config의 default-deny 정책을 따르며, tool argument는 이를 넓힐 수 없습니다.
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

protocol 오류와 tool 실행 오류를 구분해 처리하십시오. `index-not-ready`, `stale-index`, `invalid-selection`, `search-failed`, `refresh-failed`, `datasource-not-found`, `duplicates-failed`는 모델이 다음 tool call을 결정할 수 있도록 반환됩니다. `autorag.search.files`의 Everything backend는 platform/검색 실패를 구조화된 `{ ok: false, backend: "everything", reason, message }`로 반환하며, 이때에도 filesystem backend로 자동 대체하지 않습니다.

## 검증

```bash
bun run typecheck
bun run build
bunx vitest run test/mcp/server.test.ts test/mcp/stdio.test.ts
```

`test/mcp/stdio.test.ts`는 실제 stdio MCP client로 refresh, search, 파일 이름 검색, duplicates 스캔, dynamic datasource tool 노출, datasource list/get을 확인합니다.
