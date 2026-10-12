#!/usr/bin/env python3
"""Regenerate prompts/_parity/rendered.json, the expected output of rendering
every prompt with prompts/_parity/vars.json.

Python (test_prompts.py) and TypeScript (tests/prompts.test.ts) both render the
same variables and must match this file byte for byte, which is what keeps the
two loaders from drifting apart.

    python3 scripts/update_prompt_parity.py            # rewrite rendered.json
    python3 scripts/update_prompt_parity.py --check    # fail if it is stale

vars.json gains default (deliberately awkward) values for any new prompt; edit
them by hand if you want specific cases.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import prompt_loader as pl  # noqa: E402

PARITY = pl.PROMPTS_DIR / "_parity"
VARS_FILE = PARITY / "vars.json"
RENDERED_FILE = PARITY / "rendered.json"

# Characters that trip naive template engines: placeholder syntax inside a value,
# regex/replacement metacharacters, backslashes, non-ASCII, CRLF, braces.
TRICKY = "{{inner}} $& $1 \\n \\\\ é — ✓ {x} }} {{ \r\nline2\n"


def default_vars(prompt_id: str) -> dict:
    template = pl.load_prompt(prompt_id)
    values = {}
    for name, var_type in template.variables.items():
        if var_type == "string":
            values[name] = f"<{name}> {TRICKY}"
        elif var_type == "integer":
            values[name] = 7
        else:
            values[name] = True
    return values


def build() -> tuple:
    existing = json.loads(VARS_FILE.read_text("utf-8")) if VARS_FILE.exists() else {}
    all_vars, rendered = {}, {}
    for prompt_id in pl.list_prompt_ids():
        all_vars[prompt_id] = existing.get(prompt_id) or default_vars(prompt_id)
        r = pl.render_prompt(prompt_id, **all_vars[prompt_id])
        rendered[prompt_id] = {"hash": r.hash, "system": r.system, "user": r.user}
    return all_vars, rendered


def dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    all_vars, rendered = build()
    if "--check" in sys.argv:
        stale = (
            not VARS_FILE.exists() or VARS_FILE.read_text("utf-8") != dump(all_vars)
            or not RENDERED_FILE.exists() or RENDERED_FILE.read_text("utf-8") != dump(rendered)
        )
        if stale:
            print("prompts/_parity is stale: run python3 scripts/update_prompt_parity.py")
            return 1
        print("prompts/_parity is up to date")
        return 0
    PARITY.mkdir(exist_ok=True)
    VARS_FILE.write_text(dump(all_vars), "utf-8")
    RENDERED_FILE.write_text(dump(rendered), "utf-8")
    print(f"wrote {len(rendered)} prompts to {RENDERED_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
