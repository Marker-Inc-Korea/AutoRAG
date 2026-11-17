---
title: "Cloud and network folders"
description: "How AutoRAG indexes cloud sync folders, network shares, and virtual filesystems without leaving every file on disk."
---

# Cloud and network folders

This page explains how AutoRAG treats cloud sync folders, network shares, and virtual filesystems, how to connect cloud storage, and how to migrate from the removed `cloud-drive` and `spotlight` datasources.

## How cloud and network folders are treated

AutoRAG treats every location that is not a local, physical volume as a folder it must not open on its own: a cloud sync folder, a network share, or a virtual filesystem. A refresh reads each such file once — a cloud placeholder is downloaded within the budget below, a network file is read over the wire — releases the cloud placeholder it downloaded, and takes the file's dupey fingerprint on that same read. The Jikji `prepare` pass and the dupey exact-duplicate scan open file contents themselves, so both are kept away from these folders.

Every refresh detects this machine's non-physical roots from directory metadata only, never from file contents. A helper that fails (no `mount`, no `xattr`, an unmounted share) contributes no roots rather than failing the refresh. Both sides of every comparison are canonicalized, so a mount macOS reports at its real path still matches the search root the user configured.

| Root | Platform | Detected by |
| --- | --- | --- |
| File Provider mount (Google Drive, OneDrive, Dropbox, Box, …) | macOS | One directory per provider under `~/Library/CloudStorage` |
| iCloud Drive | macOS | The presence of `~/Library/Mobile Documents` |
| A File Provider–managed directory elsewhere (the iCloud-synced `~/Desktop` and `~/Documents`) | macOS | `com.apple.file-provider-domain-id` on a search root, an ancestor, or a direct child, read with `xattr` |
| Network share (SMB/NFS/AFP), WebDAV mount, FUSE volume | macOS | A mount point from `/sbin/mount` whose options lack `local` and whose type is not `autofs` |
| Cloud Files sync root (OneDrive, Dropbox, iCloud for Windows, …) | Windows | `HKCU\…\Explorer\SyncRootManager` |
| Mapped network drive (`DriveType 4`) | Windows | `Win32_LogicalDisk`, one PowerShell process per refresh |
| FUSE-compatible volume (WinFsp, Dokan, rclone) | Windows | The disk's `FileSystem` starting with `FUSE`, `WinFsp`, or `Dokan` |
| UNC search root (`\\server\share\…`) | Windows | The search root's own share prefix; `\\?\…` and `\\.\…` device paths are not shares |

**Unverified.** No Windows host with *Google Drive for desktop* installed has been measured; its virtual drive is recognized by a `GoogleDriveFS` filesystem or a `Google Drive` volume name, which is a documented guess. *Synology Drive Client* modes have not been measured on either platform: its macOS File Provider mount is found by the `~/Library/CloudStorage` listing, while a Synology virtual drive on Windows is not recognized at all.

Behaviour:

- Jikji is never run over a search root that is a cloud or network folder (or lies inside one): the root is skipped whole and reported once as the informational `jikji-cloud-root-skipped` diagnostic, and its documents are already indexed through the parsed mirror. A folder nested under a search root is excluded from the Jikji run by its relative path (`--exclude`).
- dupey never scans a search root inside such a folder and excludes a nested one by its name (`--exclude-dir`). Files in these folders join the exact-duplicate exclusion through stored fingerprints.
- A network file is read normally: no download budget and no release.
- A cloud or network file is read once per change and fingerprinted on that same read; an unchanged file is never read again.
- A root that is currently offline (an unmounted share, a signed-out provider) is never deleted: the refresh reports `search-root-unavailable` and leaves its previously indexed documents untouched.

## macOS cloud folders

A folder synced by a File Provider app (iCloud Drive, Google Drive, OneDrive under `~/Library/CloudStorage` or `~/Library/Mobile Documents`) lists every file, but a file you have not opened is a placeholder: its data is only in the cloud, and reading it downloads it. A refresh looks at the file's `st_flags` (`SF_DATALESS`) before reading.

For **iCloud Drive, Google Drive, OneDrive, and Dropbox** — all measured on macOS 26 — a refresh downloads a placeholder within the budget, parses and indexes it, then releases the local copy again with one Foundation call (`NSFileManager.evictUbiquitousItem`, run through `osascript -l JavaScript`; no helper binary, app, or entitlement is needed). The file stays in the cloud and keeps its size and modification time, so an unchanged file is never downloaded again. Files already on disk are read as before and never released.

Any **other File Provider** has no verified release path, so a placeholder is not read: it is skipped with `cloud-placeholder-skipped` (`reason` is the provider), and a previously indexed copy is kept. Make the file or folder available offline in the provider's app to index it.

Downloads are bounded per batch. A placeholder that would exceed the batch's budget ends the batch, which is handed to MinSync, and is read first in the next batch with a fresh budget, so every placeholder is indexed in the same refresh. Each downloaded file is released right after it is parsed. A refused release is retried with doubling waits (500 ms, 1 s, 2 s); when the retries are spent the file is left on disk, the document stays indexed, and one `cloud-evict-failed` records the helper's error text and the attempt count. Giving up is final: an unchanged file is not processed again. A per-file read failure is a `source-read-failed` skip, never a failed refresh.

The agent's own tools open originals too. When the indexed text is not enough, the model may read the file with `bash` or `read`, which hydrates one placeholder. AutoRAG watches the absolute paths a run's tool calls name (including the direct children of a named directory, capped at 500) and the absolute sources its retrieval results print, and releases every one that became local when the run ends. A file already on disk before the run is never released, and a release that keeps failing is logged once and never fails the answer.

```json
{ "refresh": { "cloudMaxFiles": 200, "cloudMaxBytes": 536870912, "cloudEvictAfterParse": true, "cloudEvictRetries": 3 } }
```

`cloudMaxFiles` (default `200`) and `cloudMaxBytes` (default 512 MiB) bound what one batch downloads; a batch's first placeholder is always allowed, so one large file cannot starve. `cloudEvictAfterParse: false` keeps downloaded cloud files on disk. `cloudEvictRetries` (default `3`; `0` tries once) is the number of retries after the first failed release. These keys apply on Windows too; Linux reads every file as before.

A file written on this Mac is downloaded one extra time. Its first hydrate rounds the modification time to the precision the cloud keeps (measured on macOS 26: Google Drive `+101 ns`, OneDrive `+77 ns`, iCloud `+110 ns`, Dropbox `−24 ns`), so the next refresh sees the placeholder as changed and downloads it once more. After that the value is stable and later refreshes skip the file. A file that already exists in the cloud has the cloud's rounded time from the start and is not re-downloaded.

## Windows Cloud Files

Windows sync apps (OneDrive, Dropbox, iCloud for Windows, and others built on the Cloud Files API) keep online-only files as placeholders that carry `FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS` (`0x400000`); `FILE_ATTRIBUTE_OFFLINE` is not set on them. A refresh reads the attributes of the files it is about to parse in one PowerShell process per batch, downloads placeholders within the same budget as on macOS, parses and indexes them, then releases the local copy with `CfDehydratePlaceholder` (one PowerShell process per file). The file's pin state is not changed.

Release is enabled for every Cloud Files sync root because Windows itself refuses the two unsafe cases: a file with local changes the provider has not synced (`0x80070179`) and a file pinned to stay on the device (`0x80070188`); either is reported in `cloud-evict-failed` and the file stays as it is.

Windows lets a third-party process download an online-only file only where it can show its "app is downloading files" notification. An AutoRAG refresh running without an interactive desktop (a service, or an SSH session in session 0) gets `ERROR_CLOUD_FILE_ACCESS_DENIED` (395) for every placeholder; those files are skipped with `source-read-failed` and nothing is released. Run the refresh in the desktop session, and if the notification offers to block the app, allow it (Settings, Privacy & security, Automatic file downloads).

## Connecting cloud storage

Cloud storage is connected by adding the provider's sync folder to `searchPaths`; AutoRAG keeps no separate cloud datasource and no `rclone` credential boundary. `autorag setup` reports every cloud or network folder the machine has that no `searchPaths` entry covers, with the exact command to add it.

Storage that has no desktop sync app can be mounted instead: `rclone mount <remote>: <mountpoint>` exposes a remote as a local folder, or `rclone serve webdav <remote>:` plus an OS WebDAV mount (`mount_webdav` on macOS, a mapped drive on Windows) does the same. Mounted this way the folder is a network folder: each file is read over the wire, with no download budget and nothing to release.

## Migrating from the removed cloud-drive and spotlight datasources

The rclone-backed `cloud-drive` datasource is removed and is not migrated or shimmed: because no code recognizes it, a config entry that still names it is skipped with the `unknown-datasource-skill` warning. Remove it from `datasources`, add the sync folder to `searchPaths` above, then delete the old `.autorag/datasources/<name>/` mirror data (`mirror/`, `manifest.json`, `chunks.json`) — AutoRAG no longer reads it.

The `spotlight` datasource is also removed: a config entry that names it is skipped with the `retired-datasource-skill` warning. Delete those entries and use the parsed mirror (`searchPaths`) plus fsearch name search instead.

Google Docs, Sheets, and Slides are a known gap: in a sync folder they appear only as small `.gdoc`/`.gsheet`/`.gslides` pointer files with no document content, so they are not indexed beyond their names. Whether `rclone mount` / `rclone serve webdav` exports them as readable files is unverified.

## Related pages

- [Overview](overview.md) — the pipeline and workspace layout.
- [Refresh operations](refresh-operations.md) — cloud and root diagnostics.
- [Parsing and languages](parsing-and-languages.md) — OCR and file formats.
- [Datasources overview](../datasources/overview.md) — connector-backed datasources.
