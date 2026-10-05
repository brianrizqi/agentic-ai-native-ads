#!/usr/bin/env bash
# Revision pipeline (IEEE Access resubmission). Run from agentic-ai-native-ads-main/:
#
#   bash revision/run_all.sh <stage> [model ...]
#
# Stages, in order (see revision/RUNBOOK.md for what each answers):
#   evidence    OPTIONAL per-signal evidence for train/val (OpenRouter, ~10k calls) CPU
#   prep        dedup + frozen split + shortcut probe           CPU/GPU  ~30 min (bge-m3 dedup)
#   index       retrieval indexes for 3 encoders                 GPU      ~20 min
#   knn         retrieval-only baselines + diagnostics           CPU      ~5 min
#   finetune    main adapters (reasoning_first + train-time RAG) GPU      per model
#   infer       Non-RAG and RAG on test + test_xsource           GPU      per model
#   ksweep      k = 1,3,7,10                                     GPU      ablation models
#   ablation    random / flipped / hidden labels / hidden text / near-dup / cross-lingual / MiniLM index
#   factorial   label_only, label_first, assessment_only adapters + inference
#   faithful    signal interventions on RANA
#   zeroshot    Scenario 0 floor, base models without adapters
#   latency     batch-size-1 timing on 200 articles
#   encoders    IndoBERT / XLM-R / mBERT on the same split
#   judge       LLM judge (needs OPENROUTER_API_KEY in the environment)
#   stats       all tables
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT=revision/splits/main
INDEX=revision/index/bge-m3
FMT=reasoning_first
MAIN_MODELS="gemma3-12b qwen3-8b gemma2-9b deepseek-r1-llama-8b qwen3.5-9b gemma3-4b llama3.2-1b qwen2.5-14b gemma3-1b qwen3.5-2b gemma3-270m"
ABL_MODELS="gemma3-12b qwen3-8b gemma3-1b"        # R2: k and ablations on several model sizes
FACTORIAL_MODELS="gemma3-12b qwen3-8b"             # R1-C5 / R2: same model, four output formats
# Decide BEFORE the first finetune and keep it for every run:
#   EVIDENCE=revision/splits/main/evidence.jsonl bash revision/run_all.sh finetune   -> targets carry per-signal evidence
#   (unset)                                                                         -> targets carry the four signals only
EV_ARGS=(); [[ -n "${EVIDENCE:-}" ]] && EV_ARGS=(--evidence-file "$EVIDENCE")

stage=${1:?usage: run_all.sh <stage> [model ...]}; shift || true
models=${*:-}

adapter() { echo "revision/models/$1__$2"; }
res() { echo "revision/results/$1__$2"; }

infer() {  # infer <model> <fmt> <test> <cond-name> [extra args]
  local m=$1 f=$2 t=$3 c=$4; shift 4
  python revision/infer.py --model "$(adapter "$m" "$f")" --test-file "$SPLIT/$t.jsonl" \
    --out "$(res "$m" "$f")/${t}__${c}.jsonl" "$@"
}

case $stage in
prep)
  python revision/prepare_data.py --out-dir $SPLIT --emb-cache revision/cache/dedup_bge-m3.npy
  python revision/shortcut_probe.py --split-dir $SPLIT --out revision/results/shortcut_probe.json
  ;;
evidence)
  : "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first}"
  python revision/generate_rationales.py --split-dir $SPLIT --out $SPLIT/evidence.jsonl
  ;;
index)
  python revision/build_index.py --split-dir $SPLIT --model BAAI/bge-m3 --out-dir revision/index/bge-m3
  python revision/build_index.py --split-dir $SPLIT --model sentence-transformers/all-MiniLM-L6-v2 --out-dir revision/index/minilm
  python revision/build_index.py --split-dir $SPLIT --model intfloat/multilingual-e5-large --prefix "query: " --out-dir revision/index/e5-large
  ;;
knn)
  for enc in bge-m3 minilm e5-large; do
    python revision/knn_baseline.py --index-dir revision/index/$enc --split-dir $SPLIT --out-dir revision/results/knn/$enc
  done
  ;;
finetune)
  for m in ${models:-$MAIN_MODELS}; do
    python revision/finetune.py --model "$m" --split-dir $SPLIT --index-dir $INDEX --rag --format $FMT \
      --out "$(adapter "$m" $FMT)" ${EV_ARGS[@]+"${EV_ARGS[@]}"}
  done
  ;;
infer)
  for m in ${models:-$MAIN_MODELS}; do
    for t in test test_xsource; do
      infer "$m" $FMT $t norag
      infer "$m" $FMT $t rag_k5 --rag --k 5
    done
  done
  ;;
ksweep)
  for m in ${models:-$ABL_MODELS}; do
    for k in 1 3 7 10; do infer "$m" $FMT test "rag_k$k" --rag --k "$k"; done
  done
  ;;
ablation)
  for m in ${models:-$ABL_MODELS}; do
    infer "$m" $FMT test rag_k5_random      --rag --rag-mode random
    infer "$m" $FMT test rag_k5_flipped     --rag --rag-mode flipped
    infer "$m" $FMT test rag_k5_nolabels    --rag --hide-labels
    infer "$m" $FMT test rag_k5_notext      --rag --hide-text
    infer "$m" $FMT test rag_k5_maxsim0.90  --rag --max-sim 0.90
    infer "$m" $FMT test rag_k5_lang-same   --rag --neighbor-lang same
    infer "$m" $FMT test rag_k5_lang-cross  --rag --neighbor-lang cross
    infer "$m" $FMT test rag_k5_minilm      --rag --index-dir revision/index/minilm
  done
  ;;
factorial)
  for m in ${models:-$FACTORIAL_MODELS}; do
    for f in label_only label_first assessment_only; do
      python revision/finetune.py --model "$m" --split-dir $SPLIT --index-dir $INDEX --rag --format $f \
        --out "$(adapter "$m" $f)" ${EV_ARGS[@]+"${EV_ARGS[@]}"}
      infer "$m" $f test norag
      infer "$m" $f test rag_k5 --rag --k 5
    done
  done
  ;;
faithful)
  for m in ${models:-gemma3-12b}; do
    for v in flip_each all_yes all_no; do
      infer "$m" $FMT test "rag_k5_intervene-$v" --rag --intervene $v --max-samples 500
    done
  done
  ;;
zeroshot)
  declare -A BASE=([gemma3-12b]=unsloth/gemma-3-12b-it-bnb-4bit [qwen3-8b]=unsloth/Qwen3-8B-bnb-4bit
                   [gemma2-9b]=unsloth/gemma-2-9b-it-bnb-4bit [llama3.2-1b]=unsloth/Llama-3.2-1B-Instruct-bnb-4bit)
  for m in ${models:-${!BASE[@]}}; do
    python revision/infer.py --model "${BASE[$m]}" --format $FMT --test-file $SPLIT/test.jsonl \
      --out "revision/results/zeroshot__$m/test__norag.jsonl"
  done
  ;;
latency)
  for m in ${models:-$MAIN_MODELS}; do
    infer "$m" $FMT test latency_rag_k5_bs1 --rag --k 5 --batch-size 1 --max-samples 200
  done
  ;;
encoders)
  for e in indobenchmark/indobert-base-p1 xlm-roberta-base bert-base-multilingual-cased; do
    python revision/encoder_baselines.py --model $e --split-dir $SPLIT --out-dir "revision/results/encoders__${e##*/}"
  done
  ;;
judge)
  : "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first}"
  for m in ${models:-gemma3-12b qwen3-8b gemma2-9b deepseek-r1-llama-8b}; do
    python revision/judge.py run --pred "$(res "$m" $FMT)/test__rag_k5.jsonl" --test-file $SPLIT/test.jsonl \
      --out "revision/judge/${m}__rag_k5.jsonl" --n-natural 150 --n-errors 50
  done
  python revision/judge.py export-human --judged "revision/judge/*.jsonl" --test-file $SPLIT/test.jsonl \
    --n 120 --out revision/judge/human_sheet.csv
  ;;
stats)
  # Family 1: retrieval effect for every model, in-distribution and cross-publisher (Holm over this family)
  python revision/stats.py --runs "revision/results/*__$FMT/test__norag.jsonl" "revision/results/*__$FMT/test__rag_k5.jsonl" \
    "revision/results/*__$FMT/test_xsource__norag.jsonl" "revision/results/*__$FMT/test_xsource__rag_k5.jsonl" \
    --out-dir revision/report/main --auto-pairs test__norag:test__rag_k5 --auto-pairs test_xsource__norag:test_xsource__rag_k5
  # Family 2: retrieval ablations and k sweep against the default RAG run
  args=()
  for c in rag_k5_random rag_k5_flipped rag_k5_nolabels rag_k5_notext rag_k5_maxsim0.90 rag_k5_lang-same \
           rag_k5_lang-cross rag_k5_minilm rag_k1 rag_k3 rag_k7 rag_k10; do args+=(--auto-pairs "test__rag_k5:test__$c"); done
  python revision/stats.py --runs "revision/results/*__$FMT/test__*.jsonl" --out-dir revision/report/ablation "${args[@]}"
  # Family 3: output-format factorial (same model, same data), each format vs reasoning_first
  args=()
  for m in $FACTORIAL_MODELS; do
    for f in label_only label_first assessment_only; do
      for c in norag rag_k5; do args+=(--pair "${m}__$f/test__$c:${m}__$FMT/test__$c"); done
    done
  done
  python revision/stats.py --runs "revision/results/*__*/test__rag_k5.jsonl" "revision/results/*__*/test__norag.jsonl" \
    --out-dir revision/report/factorial "${args[@]}"
  # Family 4: baselines vs RANA on the same test set
  python revision/stats.py --runs "revision/results/knn/*/test*__weighted_k5.jsonl" "revision/results/knn/*/test*__1nn.jsonl" \
    "revision/results/encoders__*/test*.jsonl" "revision/results/zeroshot__*/test__norag.jsonl" \
    "revision/results/gemma3-12b__$FMT/test*__rag_k5.jsonl" "revision/results/gemma3-12b__$FMT/test*__norag.jsonl" \
    --out-dir revision/report/baselines \
    --pair "knn/bge-m3/test__weighted_k5:gemma3-12b__$FMT/test__rag_k5" \
    --pair "encoders__xlm-roberta-base/test:gemma3-12b__$FMT/test__rag_k5" \
    --pair "encoders__indobert-base-p1/test:gemma3-12b__$FMT/test__rag_k5"
  # Faithfulness tables
  python revision/stats.py --runs "revision/results/*__$FMT/test__rag_k5_intervene-*.jsonl" --out-dir revision/report/faithfulness
  ;;
*)
  echo "unknown stage: $stage"; exit 1 ;;
esac
