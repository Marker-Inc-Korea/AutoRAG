# Refresh Diagnostics

`autorag refresh --method parsed --debug --json` reports one diagnostic per
source that cannot be mirrored. Each diagnostic keeps the opaque virtual
`source` and stable `code` separate from its human-readable `message`.

Parser diagnostics may also include `parserName` and a bounded `rootCause`.
The root cause is the first 500 characters of the parser's underlying error
message, followed by `...` when truncated. It never includes a stack trace or
the source's absolute filesystem path.

The skip codes are distinct:

- `parser-unavailable`: a previously indexed or parser-failed source no longer
  has its parser available.
- `parser-failed`: a registered parser failed while reading the source.
- `oversized`: the source exceeds the configured parse-size limit.
- `unsupported-file`: no parser is registered for the file extension.
- `duplicate-excluded`: duplicate detection intentionally excluded the source.
