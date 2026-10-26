---
name: organize
description: Rules for acting on files and folders on disk — sort, move, copy, bulk-rename, rename by content, convert formats, create folder structures, or move to the trash. Plan every target path before changing anything, never overwrite or permanently delete, confirm with a listing, and report every change as old path → new path.
license: Apache-2.0
---

# Organizing files and folders

Use this skill when the user asks you to act on files or folders — tidy or sort
a folder, move or copy files into folders, bulk or pattern rename, rename files
according to what their content says, convert formats, create folder
structures, archive, or move files to the trash. This is a file-organization
task, not a document search: the answer is the change on disk, not information
found in the files.

## How to do it

- Work with `bash` (`ls`, `mkdir`, `mv`, `cp`, and a bounded `find`). Read a
  file (with `read` or `bash`) only when its content decides the new name or
  folder, then carry the change out with `bash` or the file tools.
- Use the converter tools for format conversions, the duplicate scan when the
  request is about duplicates, and the file-name discovery tools to locate
  files by name.
- Do **not** search document contents, the user's datasources, or the web in
  this turn: content retrieval and the web tools are not part of this task. The
  turn prompt lists the exact tools to use and the ones to leave alone.

## Rules

- **Plan first.** List what you found, decide every target path, and check that
  no two targets share a name and that no target already exists before you move
  anything. Never overwrite an existing file.
- **Touch only what the request names.** When the request does not name a
  folder, ask which one instead of guessing, unless the conversation already
  says.
- **Never delete permanently** — no `rm`, `rmdir`, `unlink`, `find -delete`,
  `shred`, or PowerShell `Remove-Item`. Move what must go to the OS trash
  instead (macOS `/usr/bin/trash`, Linux `gio trash`, Windows Recycle Bin).
- **Confirm afterwards.** List the destination folder to check the changes
  really happened.
- **Finish with a report**, as your final message and with no tool call: every
  change as old path → new path (one per line), what you skipped and why, and
  anything that still needs the user. Write in the language of the request.
