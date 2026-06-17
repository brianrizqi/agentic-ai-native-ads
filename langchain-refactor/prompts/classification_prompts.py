"""
Classification Prompts for Native Ads Detection
Provides prompt templates for LLM-based classification
"""

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


# Few-shot examples for classification
CLASSIFICATION_EXAMPLES = [
    {
        "content": "Stres dan asupan makanan penuhi imunitas, ini tips dari dokter... (Biz Kompas)",
        "label": "native ads",
        "reasoning": "Konten mempromosikan tips kesehatan yang berujung pada brand/layanan tertentu, nada positif dan persuasif."
    },
    {
        "content": "Berkontribusi untuk negeri, Itenas buka 30.000 kuota vaksinasi massal... (Biz Kompas)",
        "label": "native ads",
        "reasoning": "Mempromosikan citra positif instansi/brand, bersifat satu sudut pandang (public relations)."
    },
    {
        "content": "Promo spesial BRI di HUT ke-128, diskon hingga Rp1.28 juta untuk berbagai produk perbankan.",
        "label": "native ads",
        "reasoning": "Promosi produk bank, diskon, ajakan transaksi, satu sudut pandang komersial."
    },
    {
        "content": "Presiden Jokowi menunjuk Ridwan Kamil sebagai kurator infrastruktur IKN pada hari Selasa.",
        "label": "berita murni",
        "reasoning": "Berita faktual kebijakan pemerintah, objektif, tidak ada promosi produk/citra brand."
    },
    {
        "content": "Dapatkan cashback hingga 50% untuk pembelian produk elektronik di Tokopedia hari ini!",
        "label": "native ads",
        "reasoning": "Ajakan beli dengan cashback, promosi e-commerce, nada sangat persuasif."
    }
]


# Example template for few-shot learning
EXAMPLE_TEMPLATE = """
Konten: {content}
Label: {label}
Alasan: {reasoning}
"""

example_prompt = PromptTemplate(
    input_variables=["content", "label", "reasoning"],
    template=EXAMPLE_TEMPLATE
)


# Main classification prompt template
CLASSIFICATION_PROMPT_TEMPLATE = """Anda adalah expert classifier untuk mendeteksi native advertising dalam berita Indonesia.

KARAKTERISTIK NATIVE ADS (INDIKATOR UTAMA):
1. Nada berita bersifat POSITIF atau NETRAL (tidak pernah mengkritik subjek).
2. Bahasa bersifat PERSUASIF (mengajak atau meyakinkan pembaca).
3. Mempromosikan PRODUK, LAYANAN, atau CITRA PERUSAHAAN/INSTANSI.
4. Hanya menyajikan SATU SUDUT PANDANG (one-sided).

INSTRUKSI PENTING:
- Analisis dengan hati-hati - TIDAK semua konten adalah native ads!
- Berita murni (pure news) sama umumnya dengan native ads
- Cari BUKTI JELAS promosi sebelum label sebagai native ads
- Jika ragu, pilih "berita murni"

{examples}

TUGAS:
Klasifikasikan konten berikut sebagai "native ads" atau "berita murni".

Judul: {title}
Ringkasan: {summary}

Konten:
{content}

{context}

OUTPUT FORMAT (JSON):
{{
  "label": "native ads" atau "berita murni",
  "confidence": 0.0-1.0,
  "reasoning": "penjelasan singkat dalam Bahasa Indonesia"
}}

Klasifikasi:"""


# Create few-shot prompt
few_shot_classification_prompt = FewShotPromptTemplate(
    examples=CLASSIFICATION_EXAMPLES,
    example_prompt=example_prompt,
    prefix="""Anda adalah expert classifier untuk mendeteksi native advertising.

KARAKTERISTIK NATIVE ADS (INDIKATOR UTAMA):
1. Nada berita bersifat POSITIF atau NETRAL (tidak pernah mengkritik subjek).
2. Bahasa bersifat PERSUASIF (meyakinkan pembaca).
3. Mempromosikan PRODUK, LAYANAN, atau CITRA BRAND.
4. Hanya menyajikan SATU SUDUT PANDANG (one-sided).

INSTRUKSI PENTING:
- Analisis dengan hati-hati - TIDAK semua konten adalah native ads!
- Berita murni sama umumnya dengan native ads
- Cari BUKTI JELAS promosi sebelum label sebagai native ads
- Jika ragu, pilih "berita murni"

Berikut adalah contoh klasifikasi:
""",
    suffix="""
TUGAS:
Klasifikasikan konten berikut.

Judul: {title}
Konten: {content}

OUTPUT FORMAT (JSON):
{{
  "label": "native ads" atau "berita murni",
  "confidence": 0.0-1.0,
  "reasoning": "penjelasan singkat"
}}

Klasifikasi:""",
    input_variables=["title", "content"]
)


# Simple classification prompt (without examples)
simple_classification_prompt = PromptTemplate(
    input_variables=["title", "summary", "content", "context"],
    template=CLASSIFICATION_PROMPT_TEMPLATE
)


# Explanation prompt
EXPLANATION_PROMPT_TEMPLATE = """Berikan penjelasan detail tentang klasifikasi berikut berdasarkan 4 karakteristik expert native advertising:

**ARTIKEL:**
{content}

**HASIL KLASIFIKASI:**
- Label: {label}
- Confidence: {confidence}
- Reasoning Awal: {reasoning}

**TUGAS ANDA:**
Analisis artikel berdasarkan 4 karakteristik Native Ads:

1. **Nada Positif/Netral**: Apakah artikel mengkritik subjek atau hanya positif/netral?
2. **Bahasa Persuasif**: Apakah ada kata-kata yang mengajak/meyakinkan pembaca?
3. **Mempromosikan Produk/Brand**: Apakah ada brand/produk yang dipromosikan?
4. **Satu Sudut Pandang**: Apakah artikel objektif atau hanya satu perspektif?

**FORMAT OUTPUT:**

## Analisis Detail

### 1. Nada & Tone
[Jelaskan tone artikel: positif/netral/negatif, ada kritik atau tidak]

### 2. Bahasa Persuasif
[Identifikasi kata-kata persuasif, contoh: "terbaik", "wajib", "solusi", dll]
[Quote spesifik dari artikel jika ada]

### 3. Promosi Produk/Brand
[Sebutkan brand/produk yang disebutkan]
[Jelaskan bagaimana produk dipresentasikan]

### 4. Perspektif
[Apakah artikel objektif atau one-sided?]
[Ada berbagai sudut pandang atau hanya satu?]

## Kesimpulan
[Ringkas kenapa artikel ini diklasifikasikan sebagai "{label}"]
[Sebutkan karakteristik mana yang paling kuat mendukung klasifikasi ini]

**PENTING**: Berikan evidence konkret dari artikel, jangan hanya pernyataan umum!

Penjelasan:"""

explanation_prompt = PromptTemplate(
    input_variables=["content", "label", "confidence", "reasoning", "title"],
    template="""Jawablah sebagai seorang pakar komunikasi media yang sedang berdiskusi santai namun profesional.

ANALISIS UNTUK ARTIKEL:
Judul: {title}
Isi: {content}

HASIL DETEKSI:
- Kategori: {label}
- Alasan Utama: {reasoning}

TUGAS ANDA:
Berikan narasi penjelasan kenapa artikel ini dikategorikan sebagai {label}. 
Gunakan poin-poin sederhana (Nada, Produk yang disebut, dan Sudut Pandang).

PENTING:
- DILARANG KERAS mengeluarkan format JSON.
- DILARANG KERAS mengulangi instruksi sistem seperti "Output JSON" atau "Klasifikasi:".
- Langsung berikan analisis Anda dalam Bahasa Indonesia yang baik.
"""
)


# Link discovery prompt for AI-driven crawling
DISCOVERY_PROMPT_TEMPLATE = """Anda adalah asisten riset media yang sangat teliti. Tugas Anda adalah mengidentifikasi URL berita asli DAN konten bersponsor (native ads/advertorial) dari daftar link yang diberikan.

Daftar Link dari {source}:
{links}

KRITERIA SELEKSI (WAJIB):
1. Pilih link yang merupakan ARTIKEL BERITA spesifik atau ARTIKEL ADVERTORIAL/SPONSORED.
2. SANGAT DISARANKAN memilih jika ada indikasi 'Advertorial', 'Sponsored', 'Branded Content', 'Infografis', atau 'Kerja Sama' dalam teks link.
3. JANGAN PILIH link yang mengandung kata '/tag/', '/indeks/', '/search/', atau '/author/'.
4. JANGAN PILIH link navigasi utama (Home, News, Bisnis, dll) atau link gallery foto/video saja.
5. Prioritaskan variasi antara berita murni dan konten promosi halus (native ads) agar dataset seimbang.

TUGAS:
- Pilih maksimal {limit} link terbaik.
- Kembalikan hasilnya dalam format JSON list of strings (URL saja).

CONTOH OUTPUT:
[
  "https://biz.kompas.com/read/2024/01/10/promosi-produk-a",
  "https://www.cnnindonesia.com/ekonomi/202401100912-berita-ekonomi-b"
]

HASIL:"""

discovery_prompt = PromptTemplate(
    input_variables=["source", "links", "limit"],
    template=DISCOVERY_PROMPT_TEMPLATE
)


SIMPLE_LOCAL_PROMPT_TEMPLATE = """Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".

Native Ads adalah konten yang MENGGABUNGKAN semua ciri berikut:
1. Nada positif/netral (tidak mengkritik subjek)
2. Bahasa persuasif (mengajak/meyakinkan)
3. Mempromosikan produk/brand/instansi
4. Hanya satu sudut pandang (tidak objektif)

Berita Murni:
- Bisa positif/netral/negatif
- Objektif, menyajikan berbagai sudut pandang
- Tidak mempromosikan produk/brand

Judul: {title}
Konten: {content}

{context}

Gunakan contoh-contoh di atas sebagai referensi pola penulisan yang serupa untuk menentukan label yang paling tepat.

Output (JSON):
"""

# Phase 13: MCQ Prompt for 270M stability (Gold Standard for tiny models)
MCQ_PROMPT_TEMPLATE = """Klasifikasikan berita berikut.
Pilih:
A. native ads
B. berita murni

Judul: {title}
Konten: {content}

{context}

Jawaban (A/B):
"""

# Phase 14: Reasoning-First MCQ Prompt (Standard for 270M Training)
REASONING_MCQ_PROMPT_TEMPLATE = """Klasifikasikan berita berikut.
Pilih:
A. native ads
B. berita murni

CIRI NATIVE ADS (HEURISTIK):
- Tone PR (Public Relations) atau rilis pers.
- Kutipan sepihak dari brand/CEO tanpa kritik.
- Adanya "Call to Action" halus atau info harga/promo.

Judul: {title}
Konten: {content}

{context}

Analisis:
"""

# Phase 23: Label-First MCQ for sub-500M models (more robust for micro tiers)
LABEL_FIRST_MCQ_PROMPT_TEMPLATE = """Klasifikasikan berita berikut.
Pilih Jawaban:
(A) Native Ads (Berbayar/Promosi)
(B) Berita Murni (Informatif/Objektif)

{context}

Judul: {title}
Konten: {content}

Jawaban (A/B):
"""

# Phase 24: Minimal English Zero-Shot for Micro models (<500M params)
# Designed to prevent 'multilingual collapse' seen in previous runs.
TINY_ZERO_SHOT_PROMPT_TEMPLATE = """Task: Classify news as "native ads" or "pure news".
Definition: "native ads" promote a brand/product with a positive tone. "pure news" is objective information.

Title: {title}
Content: {content}

Label (native ads/pure news):
"""

# Phase 25: Stable Compact JSON for Models with High Instruction Fade
# (Best for Gemma 270M/Llama 1B in Indonesian)
STABLE_COMPACT_JSON_TEMPLATE = """Klasifikasikan berita ini:
{context}

Judul: {title}
Konten: {content}

Output (JSON):
"""

simple_local_prompt = PromptTemplate(
    input_variables=["title", "content", "context"],
    template=SIMPLE_LOCAL_PROMPT_TEMPLATE
)


# Condensed version for small models (< 3B) to avoid instruction fade
CONDENSED_TRAINING_PROMPT_TEMPLATE = """{context}
TUGAS: Klasifikasikan berita di bawah sebagai "native ads" atau "berita murni".
INGAT: Berita peluncuran produk/teknologi tetaplah BERITA MURNI jika bahasanya informatif.

Judul: {title}
Konten: {content}

Output (JSON):
{{"reasoning": """

# Training-specific prompt template (matches inference format exactly)
TRAINING_PROMPT_TEMPLATE = """Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".

PENTING: 
- Jangan memilih "native ads" jika tidak ada BUKTI PROMOSI produk/brand yang nyata.
- Berita objektif (seperti sejarah, kriminal, atau penemuan ilmiah) adalah "berita murni".
- Jika ragu atau konten bersifat informatif belaka (pure news), pilih "berita murni".

Indikator Native Ads (Stealth/Halus):
- Tone rilis pers (PR) yang hanya menonjolkan satu sisi positif.
- Kutipan satu arah dari CEO/Juru Bicara brand tanpa penyeimbang.
- Kalimat persuasif (misal: "solusi tepat", "wajib coba", "inovasi terdepan").
- Informasi ketersediaan produk atau promo di akhir artikel.

Native Ads adalah konten yang MENGGABUNGKAN semua ciri berikut:
1. Nada positif/netral (tidak mengkritik subjek)
2. Bahasa persuasif (mengajak/meyakinkan)
3. Mempromosikan produk/brand/instansi
4. Hanya satu sudut pandang (tidak objektif)

Berita Murni:
- Bisa positif/netral/negatif
- Objektif, menyajikan berbagai sudut pandang
- Tidak mempromosikan produk/brand

Judul: {title}
Konten: {content}

{context}

Output (JSON):
{{"reasoning": "alasan singkat (max 150 karakter)", "label": "native ads" atau "berita murni", "confidence": 0.0-1.0}}
"""

# Phase 34: Ultimate Gold Standard (Reasoning-First Calibration)
# -----------------------------------------------------------------------------
# Reverting to the exact logic and format that produced 88-90% accuracy.

# Phase 88: Ultimate Gold Standard Synergy (Label-First Pivot)
# -----------------------------------------------------------------------------
# Reverting to the minimalist Anti-Bias rules that produced 88% accuracy, 
# but maintaining the Label-First anchor for drifting prevention.

# Phase 110: The Last Calibration (Target 89%+)
# -----------------------------------------------------------------------------
# Zero-Load with sharpened Ad-intent markers.

ULTIMATE_GOLD_STANDARD_TEMPLATE = """Tugas: Klasifikasikan sebagai "berita murni" (fakta/kebijakan) atau "native ads" (branding/promosi/PR).

Perisai Berita Murni (PENTING):
- Tokoh Publik/Negara: Instruksi Presiden, kunjungan Menteri, atau kebijakan makro negara.
- Peristiwa Umum: Bencana alam, kriminal, hukum, dan hasil pertandingan olahraga murni.
- PHK/Krisis: Berita tentang krisis perusahaan (PHK besar-besaran, kerugian, gugatan hukum).
- Seni/Musik: Pengumuman rilis lagu, album, atau konser tanpa nada pemujaan brand.

Aturan Emas Native Ads:
Hanya labeli "native ads" jika artikel memiliki niat mempromosikan keunggulan, citra positif produk, rilis pers korporasi, atau advertorial/PR Pemerintah Daerah (Pemkab/Pemkot) & BUMN.

Judul: {title}
Isi: {content}

{context}

JAWABAN (GUNAKAN FORMAT INI):
Analisa: (Berikan argumen logis 1-2 kalimat)
Label: (Pilih "berita murni" atau "native ads")
"""

SIMPLE_MICRO_TEMPLATE = """Tugas: Klasifikasikan sebagai "native ads" atau "berita murni".

{context}

Judul: {title}
Konten: {content}

Output JSON:
{{"alasan": """


# Phase 37: English Markdown Template (Final Llama Stability Fix)
# -----------------------------------------------------------------------------
# Llama 8B suffers from "Indonesian Instruction Collapse" when given heavy ID rules.
# This uses strong English system-commands but parses Indonesian text.

ENGLISH_MARKDOWN_TEMPLATE = """Task: Classify the following Indonesian article as "berita murni" (pure news, facts, objective reporting) or "native ads" (promotional, branding, PR marketing).

Title: {title}
Content: {content}

{context}

Format your answer EXACTLY like this:
Analysis: (1-2 sentences explaining your logic)
Label: (either "berita murni" or "native ads")
"""

# Phase 64: Bilingual Templates for Gemma/Llama Stability
# -----------------------------------------------------------------------------
# Using English instructions to prevent 'multilingual collapse' in small models.

BILINGUAL_GOLD_STANDARD_TEMPLATE = """Task: Classify as "berita murni" (facts/policy) or "native ads" (branding/promo/PR).

Definitions:
- "native ads": Content with clear "marketing intent". This includes active brand promotion, PR releases praising a company (CSR, Awards, MOU), or persuasive advertorials.
- "berita murni": Facts-first reporting. Includes macro policy, public incidents, crime, disasters, or neutral reports where a company is mentioned only as a secondary subject.

{context}

Title: {title}
Content: {content}

Output (Return JSON ONLY):
{{
  "analysis": "1-sentence marketing intent analysis",
  "label": "native ads/berita murni"
}}"""

BILINGUAL_MICRO_TEMPLATE = """Task: Professional News Audit. Classify as "berita murni" (News) or "native ads" (Marketing).

Instruction:
- Default to "berita murni" for neutral reporting on policy, crime, sports, or public figures (e.g. Jokowi).
- Label as "native ads" ONLY if there is a clear "Commercial Offer", "Product for Sale", or "Excessive Brand Praise" intended to drive revenue.
- If no specific product is being sold, it is NEWS.

{context}

Title: {title}
Content: {content}

Output JSON (analysis must be objective):
{{"analysis": "brief objective motive analysis", "label": "native ads/berita murni"}}"""

BILINGUAL_SILENT_TEMPLATE = """Task: Professional Media Audit. Classify the text below exactly as "berita murni" (General News) or "native ads" (Marketing/PR).

{context}

Title: {title}
Content: {content}

{{"label": \""""

# Phase 65: Nuclear PPL Fix (Ultra-Stable Micro)
# -----------------------------------------------------------------------------
# Designed for sub-1B models like Gemma 3 270M to force Label-First stability.

ULTRA_STABLE_MICRO_TEMPLATE = """Task: Classify as "native ads" or "berita murni".

{context}

Title: {title}
Content: {content}

Output JSON (LABEL FIRST):
{{"label": """














# Phase 35: Zero-Shot Gold Standard (For Micro Tiers 270M-500M)
# -----------------------------------------------------------------------------
# Resolving catastrophic PPL (612) by removing instruction overload.

ZERO_SHOT_GOLD_STANDARD_TEMPLATE = """Tugas: Klasifikasikan berita di bawah sebagai "native ads" atau "berita murni".

(PENTING: Gunakan format JSON di bawah.)

{context}

Judul: {title}
Konten: {content}

Output (JSON):"""

# Micro-tier training-matched templates (matches finetune.py exactly)
# These must match the templates used in finetune.py load_and_format_dataset()
# to avoid training/inference mismatch for micro-tier models.
MICRO_EN_TRAINING_TEMPLATE = """Classify the following article as 'native ads' or 'pure news'.

Native Ads combine ALL of these traits:
1. Positive/neutral tone (not critical of the subject)
2. Persuasive language (convincing/inviting)
3. Promotes a product, brand, or institution
4. Only one viewpoint (not objective)

Pure News:
- Can be positive/neutral/negative
- Objective, presents multiple viewpoints
- Does not promote products/brands

Title: {title}
Content: {content}

Output (JSON):
"""

MICRO_ID_TRAINING_TEMPLATE = """Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".

Native Ads adalah konten yang MENGGABUNGKAN semua ciri berikut:
1. Nada positif/netral (tidak mengkritik subjek)
2. Bahasa persuasif (mengajak/meyakinkan)
3. Mempromosikan produk/brand/instansi
4. Hanya satu sudut pandang (tidak objektif)

Berita Murni:
- Bisa positif/netral/negatif
- Objektif, menyajikan berbagai sudut pandang
- Tidak mempromosikan produk/brand

Judul: {title}
Konten: {content}

Output (JSON):
"""

# Phase 39: Alpaca Format Restoration (Gemma 3 270M Target 65%)
# -----------------------------------------------------------------------------
# Re-aligning the model prompt exactly with its Unsloth fine-tuning template.

# Phase 40: Minimalist Alpaca Templates (The Golden Alignment)
# -----------------------------------------------------------------------------
# Replicating the exact prompt structure from convert_dataset.py to ensure 
# the 270M model recognizes its fine-tuned patterns.

ALPACA_ID_MINIMAL_TEMPLATE = """### Instruction:
Anda adalah ahli klasifikasi media. Tugas: Klasifikasikan artikel di bawah secara tepat.

### Input:
{content}

### Petunjuk:
{context_short}

### Response:
{{"label": \""""

ALPACA_EN_MINIMAL_TEMPLATE = """### Instruction:
Task: Classify the following article.

### Input:
{content}

### Petunjuk:
{context_short}

### Response:
{{"label": \""""

# Phase 87: Advanced 8B Gold Template (Bernoulli Calibration)
# -----------------------------------------------------------------------------
# Designed to restore "Berita Murni" recall by adding mandatory news 
# safeguards for public topics (Sports, Politics, Disaster).

QWEN_HYPER_DISCRIMINATOR_V2 = """Tugas: Bertindaklah sebagai detektif iklan profesional. Ekstraksi niat promosi terselubung di balik bahasa berita yang halus.

MODE DETEKTIF (Wajib Agresif):
Klasifikasikan sebagai "native ads" jika ada SATU PUN indikasi berikut:
1. Citra Positif: Menyoroti prestasi, penghargaan, inovasi, atau kegiatan CSR (tanggung jawab sosial) sebuah perusahaan/instansi.
2. Nama Brand Berulang: Nama perusahaan atau produk disebut lebih dari 2 kali tanpa ada unsur urgensi publik yang mendasar.
3. Kerjasama/MOU: Pelaporan tentang peresmian, kerjasama, atau nota kesepahaman antar lembaga/perusahaan.
4. Nada Rilis Pers: Kalimat yang memuji, optimis, atau menggunakan kutipan pejabat perusahaan tanpa kritik.

PERISAI BERITA MURNI (Gunakan HANYA jika bersifat darurat/netral total):
- Kebijakan makro, Bencana alam, Kriminalitas murni, Hasil pertandingan olahraga, Krisis/Kerugian perusahaan (PHK/Rugi).

{context}

Judul: {title}
Isi: {content}

Output (JSON ONLY):
{{
  "analysis_keywords": ["Minimal 3 kata kunci indikator marketing"],
  "label": "native ads/berita murni"
}}"""

QWEN_GOLD_MINIMALIST_V3 = """Task: Classify the Indonesian article below into one of two categories: "native ads" or "berita murni". 

Criteria:
- "native ads": Content with promotional intent, PR releases, or corporate branding (MOU, Awards, CSR).
- "berita murni": Pure objective news about public events, disasters, or government policy.

Title: {title}
Content: {content}

Output (Return JSON ONLY):
{{
  "reason": "one sentence explanation",
  "label": "native ads/berita murni"
}}"""

ADVANCED_8B_GOLD_TEMPLATE = """Tugas: Klasifikasikan artikel di bawah sebagai "berita murni" atau "native ads".

Kriteria:
- "native ads": Konten dengan niat promosi, rilis PR, atau branding korporat (MOU, CSR, Penghargaan, Inovasi Produk).
- "berita murni": Berita objektif tentang peristiwa publik, bencana, kebijakan pemerintah, atau hukum.

{context}

Judul: {title}
Isi: {content}

Output (JSON ONLY):
{{
  "reason": "penjelasan singkat 1 kalimat",
  "label": "native ads/berita murni"
}}"""

ULTIMATE_GOLD_STANDARD_TEMPLATE = """Tugas: Klasifikasikan artikel ini sebagai "berita murni" atau "native ads".

PERISAI BERITA MURNI (News Shields):
- Jika isi berita bersifat darurat (bencana alam, krisis ekonomi negara, kriminalitas murni).
- Jika isi berita adalah hasil pertandingan olahraga, skor, atau jadwal turnamen.
- Jika isi berita adalah KEBIJAKAN MAKRO negara/instansi (Kenaikan suku bunga, kunjungan kenegaraan, peresmian fasilitas publik UNTUK rakyat).
- Jika berita melaporkan KERUGIAN atau KRISIS (rugi bersih, PHK, penurunan saham).

ATURAN EMAS NATIVE ADS:
- Jika artikel mempromosikan PENGHARGAAN, PRESTASI, atau INOVASI satu brand/lembaga.
- Jika artikel berupa rilis CSR, MOU (kerjasama korporat), atau seremoni peresmian yang bersifat branding.
- Jika artikel memuji subjek tanpa kritik atau menyajikan SATU SUDUT PANDANG yang menguntungkan subjek.

{context}

Judul: {title}
Isi: {content}

Output (Return JSON ONLY):
{{
  "reason": "Penjelasan singkat 1 kalimat mengapa ini berita murni/native ads",
  "label": "native ads/berita murni"
}}"""


# ============================================================================
# Phase 200: GEMMA 3 LARGE MODEL PROMPT (12B+)
# ============================================================================
# Winning formula from commit 198afc5 (March 25, 2026) = 98.5% accuracy.
#
# Key insight from historical analysis:
# - The March 25 RAG run used a deliberately MINIMAL prompt for Gemma
# - 4 criteria → RAG context block → article → "Output (JSON):"
# - NO elaborate shields, NO over-instruction, NO label-first prefix tricks
# - Gemma 12B has strong fine-tuned knowledge — trust it with clean context
# - The RAG examples (via {context}) did the heavy lifting for calibration
#
# This template is used when:
#   - model family = 'gemma'
#   - model_tier = 'standard' (i.e., 12B+, auto-detected from model name)
# ============================================================================

GEMMA_LARGE_TEMPLATE = """Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".

Native Ads adalah konten yang MENGGABUNGKAN semua ciri berikut:
1. Nada positif/netral (tidak mengkritik subjek)
2. Bahasa persuasif (mengajak/meyakinkan)
3. Mempromosikan produk/brand/instansi
4. Hanya satu sudut pandang (tidak objektif)

Berita Murni:
- Bisa positif/netral/negatif
- Objektif, menyajikan berbagai sudut pandang
- Tidak mempromosikan produk/brand

Gunakan contoh-contoh berikut sebagai referensi. Perhatikan bagian "Alasan" untuk memahami logika klasifikasi:
{context}

Target Berita yang Harus Diklasifikasi:
Judul: {title}
Konten: {content}

Output (JSON):
"""

