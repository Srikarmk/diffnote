"""Turn data/commits.jsonl into MLX training files and evalpair suites.

data/train.jsonl, data/valid.jsonl   chat-format examples for `mlx_lm lora` (prompt masked, message trained)
suites/commits-subject/              held-out repos, scored by chrF on the subject line
suites/commits-subject-styled/       the same items with the house style described in the prompt: the fair
                                     baseline for an untuned model, which has never seen these repos' style
suites/valid-subject/                the validation repos, for choosing a checkpoint
"""

from __future__ import annotations

import json
from pathlib import Path

SYSTEM = ("Write the git commit message for this change. Reply with the commit message only: a subject line, "
          "then, if the change needs it, a blank line and a body.")

STYLE = (SYSTEM + "\n\nHouse style: the subject is one sentence in the imperative mood, sentence case, no trailing "
         "period, usually 40 to 70 characters, naming what changed in plain words (several changes joined with "
         "commas or a semicolon). A docs-only change starts with the file, e.g. 'README: ...'. The body is plain "
         "prose wrapped at about 72 columns: what was wrong or missing, what changed, and how it was checked, "
         "with exact names and numbers. No headings, no emoji, no 'This commit'.")

SUITE = '''description = "{desc}"
checker = "chrf"
max_tokens = 400
system = """{system}"""
user = "{{diff}}"
group = "repo"
license = "MIT; diffs and messages from Srikarmk's public repos"
{options}'''


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def chat(r: dict, target: str = "message") -> dict:
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["diff"]},
                         {"role": "assistant", "content": r[target]}]}


def write_suite(folder: Path, desc: str, system: str, items: list[dict], first_line: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    opts = '[checker_options]\nline = "first"\n' if first_line else ""
    (folder / "suite.toml").write_text(SUITE.format(desc=desc, system=system, options=opts))
    with (folder / "items.jsonl").open("w") as f:
        for r in items:
            f.write(json.dumps({"id": r["id"], "repo": r["repo"], "diff": r["diff"], "gold": r["message"]}) + "\n")


def main() -> None:
    data = rows(Path("data/commits.jsonl"))
    split = {s: sorted((r for r in data if r["split"] == s), key=lambda r: (r["repo"], r["date"]))
             for s in ("train", "valid", "test")}
    # data/ trains whole messages; data/subject/ trains the subject line alone (same prompt)
    for folder, target in (("data", "message"), ("data/subject", "subject")):
        Path(folder).mkdir(parents=True, exist_ok=True)
        for s in ("train", "valid"):
            with Path(f"{folder}/{s}.jsonl").open("w") as f:
                for r in split[s]:
                    f.write(json.dumps(chat(r, target)) + "\n")
    test = split["test"]
    write_suite(Path("suites/commits-subject"), "Commit subject lines for held-out repos, chrF", SYSTEM, test, True)
    write_suite(Path("suites/commits-subject-styled"), "Commit subject lines, house style in the prompt, chrF",
                STYLE, test, True)
    write_suite(Path("suites/valid-subject"), "Commit subject lines for the validation repos, chrF", SYSTEM,
                split["valid"], True)
    print({s: len(v) for s, v in split.items()})


if __name__ == "__main__":
    main()
