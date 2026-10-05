# RUNBOOK revisi IEEE Access (Access-2026-43198)

Semua command dijalankan dari folder `agentic-ai-native-ads-main/` (bukan dari `langchain-refactor/`).
Pipeline baru ada di `revision/` dan **tidak memakai** `langchain-refactor/agents/classification_agent.py`
maupun `evaluate_model.py`. Kode lama jangan dipakai lagi untuk angka paper (lihat bagian "Kenapa pipeline baru").

## 0. Setup sekali

```bash
pip install -U sentence-transformers scikit-learn scipy openpyxl pandas requests   # di env yang sudah punya unsloth
export HF_TOKEN=hf_xxx                 # token BARU, yang lama sudah ter-hardcode di classification_agent.py
export OPENROUTER_API_KEY=sk-or-xxx    # key BARU, yang lama sempat ditempel di chat -> rotate
```

Jangan pernah lagi menaruh API key di argumen command (`--api-key ...`), karena tersimpan di history shell dan log.

## 1. Smoke test (±15 menit, jalankan dulu sebelum run besar)

```bash
python revision/prepare_data.py --out-dir revision/splits/main --emb-cache revision/cache/dedup_bge-m3.npy
python revision/build_index.py --split-dir revision/splits/main --model BAAI/bge-m3 --out-dir revision/index/bge-m3
python revision/finetune.py --model gemma3-1b --split-dir revision/splits/main --index-dir revision/index/bge-m3 \
    --rag --format reasoning_first --out revision/models/_smoke --max-train 200 --epochs 1 --eval-steps 10
python revision/infer.py --model revision/models/_smoke --test-file revision/splits/main/test.jsonl \
    --out revision/results/_smoke/test__rag_k5.jsonl --rag --max-samples 16
python revision/stats.py --runs revision/results/_smoke/test__rag_k5.jsonl --out-dir revision/report/_smoke --bootstrap 100
```

Cek `revision/models/_smoke/example_prompt.txt` (prompt + target) dan kolom `raw` di file prediksi.
Kalau output JSON-nya rapi, lanjut ke run penuh. Hapus `_smoke` setelahnya.

## 2. Run penuh

| Urutan | Command | Mesin | Perkiraan |
|---|---|---|---|
| 1 | `bash revision/run_all.sh prep` | GPU (dedup bge-m3) | 30 menit |
| 2 | *(opsional, putuskan sebelum fine-tune)* `bash revision/run_all.sh evidence` | CPU + API | beberapa jam, ±10rb call |
| 3 | `bash revision/run_all.sh index` | GPU | 20 menit |
| 4 | `bash revision/run_all.sh knn` | CPU | 5 menit |
| 5 | `bash revision/run_all.sh finetune` (atau per model: `... finetune gemma3-12b`) | GPU | 11 adapter |
| 6 | `bash revision/run_all.sh infer` | GPU | 11 model × 2 kondisi × 2 test |
| 7 | `bash revision/run_all.sh ksweep` | GPU | 3 model × 4 k |
| 8 | `bash revision/run_all.sh ablation` | GPU | 3 model × 8 kondisi |
| 9 | `bash revision/run_all.sh factorial` | GPU | 2 model × 3 adapter tambahan |
| 10 | `bash revision/run_all.sh faithful` | GPU | RANA, 500 artikel |
| 11 | `bash revision/run_all.sh zeroshot` | GPU | 4 base model |
| 12 | `bash revision/run_all.sh latency` | GPU | 11 × 200 artikel, batch 1 |
| 13 | `bash revision/run_all.sh encoders` | GPU | 3 encoder |
| 14 | `bash revision/run_all.sh judge` | API | 4 model × 200 item |
| 15 | `bash revision/run_all.sh stats` | CPU | menit |

Kalau memakai evidence, set variabelnya untuk **semua** fine-tune (utama dan factorial):
`EVIDENCE=revision/splits/main/evidence.jsonl bash revision/run_all.sh finetune`

Semua skrip bisa di-resume. `infer.py` dan `judge.py` melanjutkan dari artikel terakhir kalau dijalankan ulang
dengan `--out` yang sama.

### Padanan dengan command lama

Dulu:
```bash
python finetune.py --model gemma3-12b --dataset ../data/llm_dataset_12k_refined_full.json \
    --output ../models/gemma3-12b-native-ads-en --epochs 3 --max-steps 3000
python evaluate_model.py --model ../models/gemma3-12b-native-ads-en_merged_16bit/ --dataset ../data/test_heldout.json \
    --use-judge ... --api-key sk-or-... --num-sample 5085 --rag-vote --vectorstore-dir data/vectorstore_train_only --top-k 5
```

Sekarang:
```bash
python revision/finetune.py --model gemma3-12b --split-dir revision/splits/main --index-dir revision/index/bge-m3 \
    --rag --format reasoning_first --out revision/models/gemma3-12b__reasoning_first
python revision/infer.py --model revision/models/gemma3-12b__reasoning_first --test-file revision/splits/main/test.jsonl \
    --out revision/results/gemma3-12b__reasoning_first/test__norag.jsonl
python revision/infer.py --model revision/models/gemma3-12b__reasoning_first --test-file revision/splits/main/test.jsonl \
    --out revision/results/gemma3-12b__reasoning_first/test__rag_k5.jsonl --rag --k 5
python revision/judge.py run --pred revision/results/gemma3-12b__reasoning_first/test__rag_k5.jsonl \
    --test-file revision/splits/main/test.jsonl --out revision/judge/gemma3-12b__rag_k5.jsonl
```

Perbedaan utamanya:
- test set dikunci sebelum training, dan file training tidak pernah membaca test;
- satu adapter dipakai untuk Non-RAG maupun RAG (train-time RAG dengan dropout 50%), jadi perbandingan
  retrieval hanya berbeda di prompt;
- RAG selalu berupa prompt-injection, tidak ada lagi `--rag-vote` (post-hoc kNN override);
- tidak ada override berbasis keyword dan tidak ada hardcode prediksi;
- konfigurasi LoRA sama untuk semua model (`SHARED` di `finetune.py`). Override tercatat di `run_config.json`.

## 3. Struktur output

```
revision/splits/main/      train/val/test/test_xsource.jsonl, manifest.json, ids_hashes.csv   <- commit, jangan diubah
revision/index/<encoder>/  embedding train + cache query
revision/models/<model>__<format>/          adapter LoRA + run_config.json + example_prompt.txt
revision/results/<model>__<format>/<test>__<kondisi>.jsonl     satu baris per artikel
revision/results/knn/<encoder>/...          baseline kNN + diagnostics.json
revision/results/shortcut_probe.json        seberapa jauh label bisa ditebak dari sumber saja
revision/report/{main,ablation,factorial,baselines,faithfulness}/  CSV + report.md untuk tabel paper
```

Yang perlu dikirim balik untuk penulisan: seluruh `revision/report/`, `revision/results/shortcut_probe.json`,
`revision/results/knn/*/diagnostics.json`, `revision/splits/main/manifest.json`, dan semua `run_config.json`.

## 4. Kenapa pipeline baru (wajib dibaca sebelum menulis response letter)

1. `langchain-refactor/agents/classification_agent.py` baris 257–267 ("ACADEMIC REPLICATION PATCH")
   meng-hardcode label untuk 8 artikel tertentu (dicocokkan dari 40 karakter pertama) di jalur Gemma 12B.
   Angka apa pun yang dihasilkan lewat jalur itu, termasuk 98,50%, bukan murni output model untuk artikel tersebut.
2. Di jalur micro-tier ada override label berbasis daftar keyword (`SPORTS_KW`, `CRIME_DISASTER_KW`, ...)
   yang tidak dijelaskan di paper.
3. `reasoning` di `llm_dataset_12k_refined_full.json` hanya punya **4 kalimat unik** (satu per label × bahasa),
   dan key order-nya `{"label", "confidence", "reasoning"}`. Jadi model lama dilatih **label-first** dengan
   alasan berupa template, bukan Reasoning-First dengan penilaian empat sinyal seperti yang ditulis di paper.
   Pipeline baru memakai anotasi empat sinyal asli dari `native_ads_dataset.xlsx`.
4. Sisi English sepenuhnya berasal dari thestar.com: **semua** native ads adalah press release GlobeNewswire
   dan semua berita murni berasal dari rubrik news/sports. URL section saja sudah menebak label EN 100%.
   Di sisi Indonesia, seluruh 2.806 artikel `biz.kompas.com` berlabel native ads.
   `shortcut_probe.py` mengukur ini dan `prepare_data.py` menghapus dateline sumber dari teks.
