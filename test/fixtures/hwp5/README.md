# Real HWP5 fixture provenance

## `minimal-body-table.hwp`

- Upstream repository: `edwardkim/rhwp`
- Upstream commit: `e666360fda5de5c9a5f59d77315ba44aeca2503a`
- Upstream filename: `samples/hwp_table_test_saved.hwp`
- Immutable source URL: <https://raw.githubusercontent.com/edwardkim/rhwp/e666360fda5de5c9a5f59d77315ba44aeca2503a/samples/hwp_table_test_saved.hwp>
- SHA-256: `12110af5fb8697c90942d001ba5306e8fcf54f6655f453cb14f8436654fbee2f`
- Size: 13,824 bytes
- License: MIT; the complete pinned upstream notice is redistributed locally in [`LICENSE.rhwp`](LICENSE.rhwp) and can be verified against <https://github.com/edwardkim/rhwp/blob/e666360fda5de5c9a5f59d77315ba44aeca2503a/LICENSE>.
- License coverage: the repository-root MIT license covers the repository's software and associated documentation, including this file in the `samples/` test corpus. The upstream maintainer introduced the fixture in the pinned commit under the subject `테스트에 필요한 샘플 HWP 파일 19개 추가` ("Add 19 sample HWP files required for tests"). The pinned repository contains no separate license or exclusion for this sample.
- Selection rationale: this is a small, generic table-editing test document with no government, personal, or other third-party substantive content. It replaces the smaller `samples/hwpers_test4_complex_table.hwp` candidate because `@rhwp/core` 0.7.19 does not expose that candidate's tables through its public page-control API.

Expected parser markers:

- Body marker: `편집 탭 – 표`
- Top-level table cell marker 1: `제목`
- Top-level table cell marker 2: `담당자`
- Top-level table cell marker 3: `세부 내용`
- Rendered top-level table row: `Row 1: 제목 | 담당자 | 세부 내용`

The fixture is copied byte-for-byte from the immutable source URL. To verify it:

```sh
shasum -a 256 test/fixtures/hwp5/minimal-body-table.hwp
```
