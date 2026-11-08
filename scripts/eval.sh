#!/usr/bin/env bash
# Score one model on the held-out suites with evalpair. Every reply is cached in .evalpair/, so a rerun
# only calls the model for what is missing.
#
#   scripts/eval.sh base                 # untuned 1.5B, plain and styled prompts
#   scripts/eval.sh lora adapters/best   # the tuned 1.5B
#   scripts/eval.sh qwen3                # qwen3:8b on Ollama, styled prompt (think off)
#   scripts/eval.sh valid-base           # untuned 1.5B on the validation repos
#   scripts/eval.sh valid adapters/r16/0000136_adapters.safetensors   # one checkpoint on the validation repos
set -euo pipefail
cd "$(dirname "$0")/.."
export EVALPAIR_HOME="$PWD/.evalpair"
MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
EP=(uv run --project ../evalpair evalpair)

# mlx_lm.server applies --adapter-path only to requests for the model "default_model"; asking for the
# model by name loads it again without the adapter (every checkpoint scored the same until this).
serve() {  # serve port [--adapter-path dir]; a port of its own per model, so caches never mix
  local port=$1; shift
  if [ "${1:-}" = "--adapter-path" ] && ! grep -q "\"model\": \"$MODEL\"" "$2/adapter_config.json"; then
    echo "eval.sh: $2 was not trained on $MODEL" >&2; exit 1   # a config without model: trains mlx-lm's default
  fi
  uv run mlx_lm.server --model "$MODEL" --port "$port" "$@" >"/tmp/diffnote-server-$port.log" 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null' EXIT
  until curl -s "localhost:$port/v1/models" >/dev/null; do sleep 1; done
}

run() {  # run suite label url
  "${EP[@]}" run "suites/$1" -m default_model --backend openai --base-url "$3" --label "$2"
}

case "${1:-}" in
  base)
    serve 8081
    run commits-subject base "http://localhost:8081/v1"
    run commits-subject-styled base-styled "http://localhost:8081/v1" ;;
  lora)
    serve 8082 --adapter-path "${2:-adapters/best}"
    run commits-subject lora "http://localhost:8082/v1" ;;
  valid-base)
    serve 8091
    run valid-subject valid-base "http://localhost:8091/v1" ;;
  valid)
    # copy one checkpoint into its own adapter folder so the server loads exactly that one
    ckpt="$2"; run_dir="$(dirname "$ckpt")"
    dir="adapters/ckpt-$(basename "$run_dir")-$(basename "$ckpt" _adapters.safetensors)"
    mkdir -p "$dir" && cp "$ckpt" "$dir/adapters.safetensors" && cp "$run_dir/adapter_config.json" "$dir/"
    port=$((8100 + 10#$(basename "$ckpt" | cut -c1-7) % 800))
    serve "$port" --adapter-path "$dir"
    run valid-subject "valid-$(basename "$dir")" "http://localhost:$port/v1" ;;
  qwen3)
    "${EP[@]}" run suites/commits-subject-styled -m qwen3:8b --think off --num-ctx 8192 --label qwen3-8b-styled
    "${EP[@]}" run suites/commits-subject -m qwen3:8b --think off --num-ctx 8192 --label qwen3-8b ;;
  *) echo "usage: scripts/eval.sh base | lora [adapter] | valid <checkpoint> | qwen3" >&2; exit 2 ;;
esac
