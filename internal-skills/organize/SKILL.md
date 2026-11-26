---
name: organize
description: Rules for acting on files and folders on disk — sort, move, copy, bulk-rename, rename by content, convert formats, create folder structures, or move to the trash. Plan every target path before changing anything, never overwrite or permanently delete, confirm with a listing, and report every change as old path → new path.
license: Apache-2.0
---

# Organizing files and folders

The deliverable is the change on disk, not information retrieved from documents.

## Objective & Safety

- Inspect the target location first (read a file only when its content decides the new name or destination). Ask the user if the target folder is ambiguous.
- Plan every destination path before acting: verify no two targets collide and never overwrite an existing file.
- Follow Local File Safety: never permanently delete; move removed items to the OS trash instead.
- Use `convert_files` / `list_file_conversions` for format conversion, `scan_duplicate_documents` for duplicate checks, and file-name search tools when locating files by name.
- Verify the resulting directory after making changes, and report every change as `old path → new path` alongside anything skipped or needing user input.
