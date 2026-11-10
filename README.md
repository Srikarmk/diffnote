# diffnote

A 1.5B code model LoRA-tuned on my own commit history, run on my Mac, scored on repos it never saw. Given a
staged diff it writes the commit subject line.

On 65 commits from 6 held-out repos, the tuned model's subjects are closer to the real ones than the
untuned model's (+5.8 chrF points, 95% interval +2.6 to +9.0), still closer when the untuned model is
told the house style in its prompt (+4.0, [+0.5, +7.5]), and level with an untuned qwen3:8b, a model
five times its size (+0.7, [−2.2, +3.7]), while answering about five times faster.

```bash
git add -p
diffnote                          # prints a subject line for the staged change
git commit -e -m "$(diffnote)"    # or start the commit from it
```

## Data

469 commits from my 24 public repos: only commits I authored, no merges, lockfiles and binaries left out
of the diff, `Co-Authored-By` trailers stripped. The model sees the `--stat` summary and then the patch,
cut at 6,000 characters on a line boundary (about 1,700 tokens), so every changed file is named even when
the patch is cut.

The split is by repo, not by commit:

| Split | Commits | Repos |
|---|---|---|
| train | 363 | 14, mostly duskline (220) and Keystone (77) |
| validation | 41 | region-aware-gp, groupthink, format-tax, quant-cliff |
| test | 65 | Gitgotchi, whygit, council-votes, hearth, bostonmove, evalpair |

Every choice (learning rate, target, checkpoint) was made on the validation repos. The test repos were
scored once, at the end.

One thing to know about the data: most of these messages were written with an AI coding assistant in the
loop, so what the model learns is the house style of these repos, not hand-typed prose.

## Method

- **Model:** Qwen2.5-Coder-1.5B-Instruct (Apache 2.0), bf16, LoRA rank 16 on every attention and MLP
  projection in all 28 layers (18.5M trainable parameters, 1.2%), with
  [MLX](https://github.com/ml-explore/mlx-lm) on an M5 Pro laptop. Prompt tokens are masked from the loss.
- **Target:** the subject line. Configuration in [`lora-subject.yaml`](lora-subject.yaml): learning rate
  2e-5 with cosine decay, batch 2 with 2 accumulation steps; validation loss bottomed out at step 180 of 272,
  and that checkpoint is the one tested.
- **Scoring:** [evalpair](https://github.com/Srikarmk/evalpair) with its `chrf` checker (character n-gram
  F-score, identical to sacreBLEU's chrF) on the first line of the reply against the real subject.
  Temperature 0. Comparisons are paired by commit, with bootstrap intervals over commits, stratified by repo.
- **Baselines:** the same 1.5B untuned, with the plain prompt and with a prompt that spells out the house
  style; and qwen3:8b on Ollama (thinking off) with both prompts.

## Results

Held-out test repos, 65 commits:

| Model | chrF, subject | Paired vs tuned 1.5B | Seconds per commit |
|---|---|---|---|
| Qwen2.5-Coder-1.5B, untuned | 20.2% [17.4, 23.1] | tuned +5.8 pp [+2.6, +9.0] | 2.70 |
| … with the house style in the prompt | 21.9% [19.4, 24.8] | tuned +4.0 pp [+0.5, +7.5] | 1.51 |
| qwen3:8b, untuned, house style in the prompt | 25.3% [22.9, 28.0] | tuned +0.7 pp [−2.2, +3.7] | 4.95 |
| qwen3:8b, untuned, plain prompt | 25.6% [23.2, 28.2] | tuned +0.4 pp [−2.6, +3.6] | n/a |
| **Qwen2.5-Coder-1.5B, LoRA-tuned** | **26.0% [22.6, 29.4]** | | **0.94** |

Times are per commit, on the same laptop, one model at a time; the 1.5B models ran in MLX and qwen3:8b in
Ollama, so the timing comparison is between setups, not just model sizes.

What the numbers look like:

```
GOLD    Stop tracking build artifacts, bytecode and local pet state
base    fix: remove state.db and .gitignore files
tuned   Remove build artifacts, state.db, and .gitignore
8B      Remove old build artifacts and egg-info files

GOLD    Add gitgotchi haunt, a live pane that reacts to commits
base    fix(haunt): add live pane for pet reactions
tuned   Add haunt: live pane where the pet reacts to every commit
8B      feat(cli): add haunt command to show pet reactions to commits

GOLD    0.1.2: MCP registry metadata (server.json and the README ownership marker)
base    fix: update version in server.json and pyproject.toml
tuned   Add MCP server and release notes
8B      Add MCP registry support and update version to 0.1.2
```

The untuned 1.5B puts a conventional-commit `fix:` on nearly everything and sometimes lists every file;
the tuned one writes in the repos' own form and length (52 characters on average; the real subjects
average 62). It is also wrong in its own way: the last example names the wrong change. Per commit, the 8B
beats it 39 times and loses 26, even though the means are level.

## What didn't work, and the mistakes on the way

- **Training on whole messages (subject and body) didn't help.** The best checkpoint matched the untuned
  model on the validation repos (22.5% vs 22.0%), and longer training made subjects shorter and more
  generic (14.1% by step 204, mostly "Add <file>"). With 363 examples and per-token loss still above 2, the
  bodies were too varied to learn. Training on the subject alone went from 22.0% to 25.0% on validation,
  and that is the model above.
- **Learning rate 1e-4 diverged** (training loss 2.8 to 4.2 in 60 steps). mlx-lm's LoRA `scale` of 20
  multiplies the update, so its expected learning rate is around 1e-5; 2e-5 trained cleanly.
- **The first checkpoint comparison was void.** mlx_lm.server applies `--adapter-path` only to requests
  for the model `default_model`; asking for the model by name loaded it without the adapter, and four
  different checkpoints scored exactly the same. [`scripts/eval.sh`](scripts/eval.sh) now always asks
  for `default_model`, and refuses an adapter trained on a different base model.
- **One run trained the wrong model.** A copied config lost its `model:` line, and mlx-lm fell back to its
  default (Qwen3-0.6B) without a warning. Its run header did name the model, after downloading it, and I
  missed it. Nothing from that run is used.
- **The untuned models' replies were mis-scored at first.** They often open a code fence and run out of
  tokens before closing it, and the subject was read as "```markdown". evalpair's chrf checker now drops an
  unclosed opening fence too; all numbers above are after the fix.

## Limits

- chrF measures overlap with the real subject, not whether a subject is right. A reworded but accurate
  subject scores low, and some real subjects say what the diff can't ("FINAL: the noise-model repair fails
  too"), so no model gets near 100%. It is a fair comparison between models, not a quality grade.
- 65 test commits from 6 repos, one seed, one training run per setting. The intervals are over commits;
  they say nothing about variation between training runs.
- The training data is dominated by one game repo. The tuned model still transfers to Python tools and
  research repos, but more varied history would likely help most.

## Reproduce

The scoring script expects a clone of [evalpair](https://github.com/Srikarmk/evalpair) next to this
folder (`../evalpair`): the `chrf` checker is on its main branch and not yet in a PyPI release.

```bash
uv sync
uv run python -m diffnote.data       # data/commits.jsonl from the local clones (paths in data.py)
uv run python -m diffnote.prepare    # training files and evalpair suites
uv run mlx_lm.lora -c lora-subject.yaml   # about 30 minutes on an M5 Pro
scripts/eval.sh valid adapters/subj/0000180_adapters.safetensors
cp -r adapters/ckpt-subj-0000180 adapters/best
scripts/eval.sh base && scripts/eval.sh lora && scripts/eval.sh qwen3
```

Every model reply is in `.evalpair/runs/`, and the training and evaluation logs are in `results/`.

## Layout

```
src/diffnote/data.py      commits -> (diff, message) pairs, repo split
src/diffnote/prepare.py   MLX chat files and evalpair suites
src/diffnote/cli.py       the diffnote command
suites/                   evalpair suites (test: commits-subject, commits-subject-styled; valid-subject)
lora.yaml                 whole-message run (did not help); lora-subject.yaml: the model above
results/                  training and evaluation logs
```

## License

MIT. The diffs and messages come from my own public repos.
