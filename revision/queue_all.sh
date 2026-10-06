#!/usr/bin/env bash
# Runs every remaining GPU stage back to back, one job at a time, in priority order:
# main table (10 models) -> output-format factorial -> retrieval ablations -> encoder
# baselines -> k sweep -> zero-shot -> faithfulness -> latency -> stats.
#
# Jobs never share the GPU, so memory stays safe and latency figures stay valid.
# Each model/stage runs on its own: a failure is logged and the queue moves on.
# It first waits for any revision/infer.py or revision/finetune.py already running.
#
#   nohup bash revision/queue_all.sh > revision/logs/queue.log 2>&1 &
#   cat revision/logs/queue.log          # START / OK / FAIL per job
cd "$(dirname "$0")/.."
LOG=revision/logs
mkdir -p "$LOG"

run() {
  local name=$1; shift
  echo "[$(date '+%F %T')] START $name"
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] OK    $name"
  else
    echo "[$(date '+%F %T')] FAIL  $name (see $LOG/$name.log)"
  fi
}

while pgrep -f "revision/(infer|finetune)\.py" > /dev/null; do sleep 300; done
echo "[$(date '+%F %T')] GPU free, queue starts"

run stats_rana bash revision/run_all.sh stats

REST="qwen3-8b gemma2-9b deepseek-r1-llama-8b qwen3.5-9b gemma3-4b llama3.2-1b qwen2.5-14b gemma3-1b qwen3.5-2b gemma3-270m"
for m in $REST; do
  run "ft_$m" bash revision/run_all.sh finetune "$m"
  if [ -f "revision/models/${m}__reasoning_first/run_config.json" ]; then
    run "infer_$m" bash revision/run_all.sh infer "$m"
  fi
done
run stats_main bash revision/run_all.sh stats

for m in gemma3-12b qwen3-8b; do run "factorial_$m" bash revision/run_all.sh factorial "$m"; done
for m in gemma3-12b qwen3-8b gemma3-1b; do run "ablation_$m" bash revision/run_all.sh ablation "$m"; done
run encoders bash revision/run_all.sh encoders
for m in gemma3-12b qwen3-8b gemma3-1b; do run "ksweep_$m" bash revision/run_all.sh ksweep "$m"; done
run zeroshot bash revision/run_all.sh zeroshot
run faithful bash revision/run_all.sh faithful
run latency bash revision/run_all.sh latency
run stats_all bash revision/run_all.sh stats

echo "[$(date '+%F %T')] QUEUE DONE"
