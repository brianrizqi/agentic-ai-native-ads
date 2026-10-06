#!/usr/bin/env bash
# Runs the remaining GPU stages back to back, one job at a time, in priority order,
# and never starts a job that cannot finish before the reservation ends.
#
#   DEADLINE="2026-10-09 18:00" nohup bash revision/queue_all.sh > revision/logs/queue.log 2>&1 &
#   cat revision/logs/queue.log          # START / OK / FAIL / SKIP per job
#
# DEADLINE is when the server reservation ends. BUFFER_H (default 4) hours are kept
# free at the end for backing up results. Estimates (hours, A100 40 GB) come from the
# measured RANA runs: 12B fine-tune 5.9 h, 12B inference 2.1 s/article.
# Jobs never share the GPU, so memory stays safe and timing figures stay valid.
cd "$(dirname "$0")/.."
LOG=revision/logs
mkdir -p "$LOG"
: "${DEADLINE:?set DEADLINE, e.g. DEADLINE=\"2026-10-09 18:00\"}"
DEADLINE_S=$(date -d "$DEADLINE" +%s)
BUFFER_H=${BUFFER_H:-4}
SPLIT=revision/splits/main
INDEX=revision/index/bge-m3

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

main_model() {  # fine-tune + Non-RAG/RAG inference on test and withheld publisher
  local m=$1 ft=$2 inf=$3
  run "ft_$m" "$ft" bash revision/run_all.sh finetune "$m"
  if [ -f "revision/models/${m}__reasoning_first/run_config.json" ]; then
    run "infer_$m" "$inf" bash revision/run_all.sh infer "$m"
  fi
}

factorial_format() {  # one extra output format on the factorial model, test partition only
  local m=$1 f=$2 ft=$3
  local out="revision/models/${m}__$f" res="revision/results/${m}__$f"
  run "ft_${m}_$f" "$ft" python revision/finetune.py --model "$m" --split-dir $SPLIT --index-dir $INDEX \
      --rag --format "$f" --out "$out"
  if [ -f "$out/run_config.json" ]; then
    run "infer_${m}_$f" 2 bash -c "python revision/infer.py --model $out --test-file $SPLIT/test.jsonl --out $res/test__norag.jsonl && \
      python revision/infer.py --model $out --test-file $SPLIT/test.jsonl --out $res/test__rag_k5.jsonl --rag --k 5"
  fi
}

while pgrep -f "revision/(infer|finetune)\.py" > /dev/null; do sleep 300; done
echo "[$(date '+%F %T')] GPU free, queue starts (deadline $DEADLINE, buffer ${BUFFER_H} h)"
run stats_rana 1 bash revision/run_all.sh stats

# 1. Retrieval controls on RANA (R1-C2, R1-C3, R1-C4)
run ablation_gemma3-12b 9 bash revision/run_all.sh ablation gemma3-12b
# 2. Faithfulness on RANA (R1-C5)
run faithful_gemma3-12b 2 env FAITH=flip_each bash revision/run_all.sh faithful gemma3-12b
# 3. Encoder baselines on the same split (R1-C9)
run encoders 2 bash revision/run_all.sh encoders
# 4. Output-format factorial on one model (R1-C5, R2-4): Gemma 3 4B, all four formats
main_model gemma3-4b 4 4
factorial_format gemma3-4b label_first 4
factorial_format gemma3-4b label_only 4
factorial_format gemma3-4b assessment_only 4
# 5. Further models for the main table (R1-C10, R1-C11)
main_model qwen3-8b 5 4
main_model gemma3-1b 3 3
# 6. Retrieval depth on two model sizes (R2-3)
run ksweep_gemma3-12b 4 env KS="1 3 10" bash revision/run_all.sh ksweep gemma3-12b
run ksweep_gemma3-1b 2 env KS="1 3 10" bash revision/run_all.sh ksweep gemma3-1b
# 7. Optional, only if time remains
main_model llama3.2-1b 3 3
run zeroshot_gemma3-12b 2 bash revision/run_all.sh zeroshot gemma3-12b
main_model deepseek-r1-llama-8b 5 4

run stats_all 1 env FACTORIAL_MODELS=gemma3-4b bash revision/run_all.sh stats
echo "[$(date '+%F %T')] QUEUE DONE"
