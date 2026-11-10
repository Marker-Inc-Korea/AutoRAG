# AutoRAG for Hermes Agent - 90 second demo.
# Needs: vhs, an isolated Hermes home with the plugin enabled and its gateway running, and the
# synthetic corpus from make-fixtures.py (see README.md in this directory).
Output demo.mp4
Output demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1500
Set Height 900
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 18ms
Set PlaybackSpeed 1

Hide
Type "export PS1='$ ' && alias hermes='./hermes-demo-env.sh ${HERMES_BIN:-hermes}' && clear"
Enter
Show

Type "# Hermes + AutoRAG plugin: files, Slack, Discord and Telegram, all local"
Enter
Sleep 1.5s

Type "hermes plugins show autorag | head -9"
Enter
Sleep 3s

Type "hermes -t autorag -z 'In my Hanbit contract PDF, what are the unit price and the termination notice? Cite the file.'"
Enter
Wait@90s /^\$ *$/
Sleep 3s

Type "clear"
Enter
Type "hermes -t autorag -z 'Slack, Discord and Telegram all discuss the Hanbit renewal. Compare them with the contract PDF: list each mismatch and its source, briefly.'"
Enter
Wait@120s /^\$ *$/
Sleep 5s

Type "clear"
Enter
Type "hermes -t autorag -z 'Any duplicate files in my documents? Do not delete anything.'"
Enter
Wait@90s /^\$ *$/
Sleep 4s
