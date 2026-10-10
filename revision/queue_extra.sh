#!/usr/bin/env bash
# Extra runs after the main queue: RANA with two more seeds (R1 Q17) and the output-format
# factorial repeated on Gemma 3 12B (R1-C5, R2-4). Same rules as queue_all.sh: one job at a
# time, never starts a job that cannot finish before DEADLINE, safe to rerun (finished
# adapters are skipped, fine-tunes resume from checkpoints, inference from the last article).
#
#   BUFFER_H=0 DEADLINE="2026-10-13 19:30" nohup bash revision/queue_extra.sh >> revision/logs/queue_extra.log 2>&1 &
cd "$(dirname "$0")/.."
LOG=revision/logs
mkdir -p "$LOG"
: "${DEADLINE:?set DEADLINE, e.g. DEADLINE=\"2026-10-13 19:30\"}"
DEADLINE_S=$(date -d "$DEADLINE" +%s) || exit 1
BUFFER_H=${BUFFER_H:-0}
SPLIT=revision/splits/main
INDEX=revision/index/bge-m3
M=gemma3-12b

run() {  # run <name> <estimated hours> <command...>
  local name=$1 est=$2; shift 2
  local now; now=$(date +%s)
  if (( now + (est + BUFFER_H) * 3600 > DEADLINE_S )); then
    echo "[$(date '+%F %T')] SKIP  $name (needs ~${est} h, not enough time before $DEADLINE)"
    return
  fi
  echo "[$(date '+%F %T')] START $name (~${est} h)"
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] OK    $name"
  else
    echo "[$(date '+%F %T')] FAIL  $name (see $LOG/$name.log)"
  fi
}

infer4() {  # Non-RAG and RAG on the test partition, optionally on the withheld publisher too
  local model=$1 res=$2 sets=$3
  for t in $sets; do
    python revision/infer.py --model "$model" --test-file "$SPLIT/$t.jsonl" --out "$res/${t}__norag.jsonl" || return 1
    python revision/infer.py --model "$model" --test-file "$SPLIT/$t.jsonl" --out "$res/${t}__rag_k5.jsonl" --rag --k 5 || return 1
  done
}

adapter_job() {  # adapter_job <tag> <format> <seed> <infer hours> <test sets>
  local tag=$1 fmt=$2 seed=$3 inf=$4 sets=$5
  local out="revision/models/${M}__$tag" res="revision/results/${M}__$tag"
  if [ -f "$out/run_config.json" ]; then
    echo "[$(date '+%F %T')] DONE  ft_${M}_$tag (adapter exists)"
  else
    run "ft_${M}_$tag" 6 python revision/finetune.py --model $M --split-dir $SPLIT --index-dir $INDEX \
        --rag --format "$fmt" --seed "$seed" --out "$out"
  fi
  if [ -f "$out/run_config.json" ]; then
    run "infer_${M}_$tag" "$inf" infer4 "$out" "$res" "$sets"
  fi
}

while pgrep -f "revision/(infer|finetune)\.py" > /dev/null; do sleep 300; done
echo "[$(date '+%F %T')] GPU free, extra queue starts (deadline $DEADLINE, buffer ${BUFFER_H} h)"

# 1. RANA with two more seeds, both test sets (seed 3407 is the main run)
adapter_job reasoning_first_s42   reasoning_first 42   7 "test test_xsource"
adapter_job reasoning_first_s2024 reasoning_first 2024 7 "test test_xsource"
# 2. Output-format factorial on Gemma 3 12B, test partition (reasoning_first = main run)
adapter_job label_first     label_first     3407 4 "test"
adapter_job label_only      label_only      3407 3 "test"
adapter_job assessment_only assessment_only 3407 4 "test"

# Stats: seeds side by side; factorial on both model sizes
python revision/stats.py --runs "revision/results/${M}__reasoning_first*/test__norag.jsonl" \
  "revision/results/${M}__reasoning_first*/test__rag_k5.jsonl" \
  "revision/results/${M}__reasoning_first*/test_xsource__norag.jsonl" \
  "revision/results/${M}__reasoning_first*/test_xsource__rag_k5.jsonl" \
  --out-dir revision/report/seeds > "$LOG/stats_seeds.log" 2>&1
FACTORIAL_MODELS="gemma3-4b gemma3-12b" bash revision/run_all.sh stats > "$LOG/stats_extra.log" 2>&1
echo "[$(date '+%F %T')] EXTRA QUEUE DONE"
