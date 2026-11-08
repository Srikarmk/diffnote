"""diffnote: suggest a commit message for the staged change, written by the tuned model on this Mac.

    diffnote                 # message for `git diff --staged` in the current repo
    diffnote --base          # the same, from the untuned model, for comparison
    git commit -e -m "$(diffnote)"

Nothing leaves the machine: the model and adapter are loaded from local files with MLX.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import staged_diff
from .prepare import SYSTEM

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER = Path(__file__).resolve().parents[2] / "adapters" / "best"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="diffnote", description="Suggest a commit message for the staged change.")
    p.add_argument("--base", action="store_true", help="use the untuned model")
    p.add_argument("--adapter", default=str(ADAPTER), help="LoRA adapter folder")
    p.add_argument("--max-tokens", type=int, default=400)
    a = p.parse_args(argv)

    diff = staged_diff()
    if not diff:
        print("diffnote: nothing staged (git add first)", file=sys.stderr)
        return 1
    from mlx_lm import generate, load  # slow import; only once there is something to describe

    model, tok = load(MODEL, adapter_path=None if a.base else a.adapter)
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": diff}],
                                     add_generation_prompt=True, tokenize=False)
    print(generate(model, tok, prompt, max_tokens=a.max_tokens).strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
