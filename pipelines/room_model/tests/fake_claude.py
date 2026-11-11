#!/usr/bin/env python3
"""Stand-in for the `claude` binary: logs its call and replays a canned answer (selected through environment variables).

FAKE_CLAUDE_LOG    file receiving one JSON line {argv, cwd, stdin} per call
FAKE_CLAUDE_ANSWER file whose text becomes the `result` field of the JSON envelope
FAKE_CLAUDE_RAW    when set, the answer file is printed as is, without the envelope
FAKE_CLAUDE_EXIT   when set, print an error on stderr and exit with this code
"""
import json
import os
import sys

prompt = sys.stdin.read()
log = os.environ.get("FAKE_CLAUDE_LOG")
if log:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": prompt, "files": sorted(os.listdir("."))}) + "\n")
if os.environ.get("FAKE_CLAUDE_EXIT"):
    print("fake claude: authentication failed", file=sys.stderr)
    sys.exit(int(os.environ["FAKE_CLAUDE_EXIT"]))
with open(os.environ["FAKE_CLAUDE_ANSWER"], encoding="utf-8") as fh:
    text = fh.read()
if os.environ.get("FAKE_CLAUDE_RAW"):
    print(text)
else:
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": text, "duration_ms": 1}))
