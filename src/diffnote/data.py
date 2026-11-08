"""Build the commit dataset: (diff, message) pairs from my own public repos.

Only repos owned by Srikarmk and public on GitHub, and only commits authored as Srikarmk, so the data and
anything trained on it can be published. Merge commits, empty diffs and pure lockfile/binary changes are
dropped. Trailers (Co-Authored-By and friends) are stripped from the message: they are boilerplate, not style.

Split by repo, not by commit: the test repos are never seen in training, so the score says whether the model
learned to describe a change, not whether it memorised a project's vocabulary.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path.home() / "Programming"

# folder -> GitHub repo; all public and owned by Srikarmk (checked with `gh repo view`)
REPOS = {
    "Aurora-GP": "region-aware-gp", "Gitgotchi": "Gitgotchi", "Keystone": "Keystone",
    "MLOps-Lab2": "MLOps-Lab2", "MLOps-Lab3": "MLOps-Lab3", "Silent-RAG-Failures": "Silent-RAG-Failures",
    "Srikarmk-profile": "Srikarmk", "bostonmove": "bostonmove", "council-votes": "council-votes",
    "datasniff": "datasniff", "evalpair": "evalpair", "experiment-forensics": "experiment-forensics",
    "fantasy-racer": "duskline", "format-tax": "format-tax", "groupthink": "groupthink", "hearth": "hearth",
    "llm-judge-bias": "llm-judge-bias", "long-context-recall": "long-context-recall", "memory-rot": "memory-rot",
    "prompt-brittleness": "prompt-brittleness", "quant-cliff": "quant-cliff", "rag-poisoning": "rag-poisoning",
    "tool-call-reliability": "tool-call-reliability", "whygit": "whygit",
}

# never seen in training or validation
TEST = {"Gitgotchi", "whygit", "council-votes", "hearth", "bostonmove", "evalpair"}
# for early stopping and choosing the checkpoint; also unseen in training
VALID = {"Aurora-GP", "groupthink", "format-tax", "quant-cliff"}

SKIP_FILES = re.compile(
    r"(^|/)(package-lock\.json|pnpm-lock\.yaml|yarn\.lock|uv\.lock|poetry\.lock|Cargo\.lock)$"
    r"|\.(png|jpe?g|webp|gif|ico|pdf|glb|gltf|bin|wav|mp3|ogg|woff2?|ttf|sqlite|parquet|pkl|zip)$"
)
TRAILER = re.compile(r"^(Co-Authored-By|Signed-off-by|Generated-by|Reviewed-by):", re.I)
DIFF_CHARS = 6000   # about 1,700 tokens of patch; the file summary always comes first, so nothing is hidden


@dataclass
class Commit:
    id: str            # repo:sha
    repo: str
    sha: str
    date: str
    split: str
    files: list[str]
    diff: str          # stat summary + truncated patch
    message: str       # subject + body, trailers removed
    subject: str


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT / repo), *args], capture_output=True, text=True,
                          check=True, errors="replace").stdout


def clean_message(raw: str) -> str:
    lines = [l for l in raw.strip().splitlines() if not TRAILER.match(l.strip())]
    return "\n".join(lines).strip()


def render(stat: str, patch: str) -> str:
    """The model's input: the file summary, then the patch cut at DIFF_CHARS on a line boundary."""
    if len(patch) > DIFF_CHARS:
        patch = patch[:DIFF_CHARS].rsplit("\n", 1)[0] + "\n[... diff truncated ...]"
    return f"{stat.rstrip()}\n\n{patch}".strip()


def diff_for(repo: str, sha: str) -> tuple[list[str], str]:
    names = [n for n in git(repo, "show", "--format=", "--name-only", sha).splitlines() if n]
    keep = [n for n in names if not SKIP_FILES.search(n)]
    if not keep:
        return names, ""
    stat = git(repo, "show", "--format=", "--stat=100", sha, "--", *keep)
    patch = git(repo, "show", "--format=", "--no-color", "-U2", sha, "--", *keep)
    return names, render(stat, patch)


def staged_diff(cwd: str = ".") -> str:
    """The staged change in the repo at `cwd`, rendered exactly as the training inputs were."""
    run = lambda *a: subprocess.run(["git", "-C", cwd, *a], capture_output=True, text=True, check=True,
                                    errors="replace").stdout
    keep = [n for n in run("diff", "--cached", "--name-only").splitlines() if n and not SKIP_FILES.search(n)]
    if not keep:
        return ""
    return render(run("diff", "--cached", "--stat=100", "--", *keep),
                  run("diff", "--cached", "--no-color", "-U2", "--", *keep))


def commits(repo: str) -> list[Commit]:
    split = "test" if repo in TEST else "valid" if repo in VALID else "train"
    out = []
    log = git(repo, "log", "--no-merges", "--author=Srikarmk", "--format=%H%x1f%aI%x1f%B%x1e")
    for rec in log.split("\x1e"):
        if not rec.strip():
            continue
        sha, date, body = rec.strip().split("\x1f", 2)
        message = clean_message(body)
        files, diff = diff_for(repo, sha)
        if not diff or not message:
            continue
        out.append(Commit(f"{REPOS[repo]}:{sha[:10]}", REPOS[repo], sha, date, split, files, diff, message,
                          message.splitlines()[0]))
    return out


def build(out: Path) -> dict[str, int]:
    rows = [c for repo in REPOS for c in commits(repo)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c in rows:
            f.write(json.dumps(asdict(c)) + "\n")
    counts: dict[str, int] = {}
    for c in rows:
        counts[c.split] = counts.get(c.split, 0) + 1
    return counts


if __name__ == "__main__":
    print(build(Path("data/commits.jsonl")))
