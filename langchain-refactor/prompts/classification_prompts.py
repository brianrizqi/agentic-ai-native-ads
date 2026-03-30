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

# Phase 101: The Recursive Gold Reset (Target 89%+)
# -----------------------------------------------------------------------------
# Rebalancing from the most stable foundation (Phase 95).

ULTIMATE_GOLD_STANDARD_TEMPLATE = """Analisis artikel ini sebagai Media Analyst yang sangat teliti. Berikan klasifikasi "berita murni" atau "native ads".

[KRITERIA]:
- BERITA MURNI: Laporan objektif tentang politik, kebijakan publik, diplomatik, tragedi, statistik industri, atau krisis korporasi (PHK/Hukum).
- NATIVE ADS: Konten dengan tujuan promosi, branding, atau citra positif. Termasuk pelaporan pencapaian sepihak (Awards/MOU), Press Release bisnis, promosi produk/jasa, atau edukasi/tips yang mengarah pada brand tertentu.

{context}

TUGAS:
Judul: {title}
Konten: {content}

PENTING: Tentukan label terlebih dahulu sebelum memberikan alasan.
Output (JSON):
{{"label": "berita murni/native ads", "alasan": "Analisis singkat (Indonesia)"}}"""

SIMPLE_MICRO_TEMPLATE = """Tugas: Klasifikasikan sebagai "native ads" atau "berita murni".

{context}

Judul: {title}
Konten: {content}

Output JSON:
{{"alasan": """


# Phase 64: Bilingual Templates for Gemma/Llama Stability
# -----------------------------------------------------------------------------
# Using English instructions to prevent 'multilingual collapse' in small models.

BILINGUAL_GOLD_STANDARD_TEMPLATE = """Task: Classify the following Indonesian news article as "native ads" or "berita murni" (pure news).

ANTI-BIAS RULES (IMPORTANT):
1. PURE NEWS STAYS PURE NEWS: Do not label as ads just because it mentions a brand or price.
2. TRAGEDY & SOCIAL: Accidents, deaths, public policies, and natural disasters are 100% PURE NEWS.
3. SPORTS: Match scores, athlete injuries, and tournament results are PURE NEWS.
- NATIVE ADS: Only if the content is purely commercial promotion for a specific brand (PR release style).

{context}

ARTICLE TO CLASSIFY:
Title: {title}
Content: {content}

Output (JSON):
{{"alasan": """

BILINGUAL_MICRO_TEMPLATE = """Task: Classify as "native ads" or "berita murni".

{context}

Title: {title}
Content: {content}

Output JSON:
{{"alasan": """

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

# Phase 83: Micro-Heuristic Prompt (The Stability Anchor)
# -----------------------------------------------------------------------------
# Designed for 270M models to prevent "Context Drowning." It provides 
# ultra-compact heuristics and uses recency bias by putting the task 
# at the very bottom.

MICRO_HEURISTIC_PROMPT = """Tugas: Klasifikasikan berita di bawah sebagai "native ads" atau "berita murni".

[ATURAN ANTI-BIAS (PENTING)]:
1. TRAGEDI & SOSIAL: Kecelakaan, kematian, dan bencana = BERITA MURNI 100%.
2. OLAHRAGA & KEBIJAKAN: Skor, atlet, dan info pemerintah = BERITA MURNI.
3. NATIVE ADS: Hanya jika isinya murni PROMOSI produk/brand (Advertorial).

{context}

Judul: {title}
Konten: {content}

Pilih Jawaban:
A. native ads
B. berita murni

Jawaban (A/B):"""

# Phase 87: Advanced 8B Gold Template (Bernoulli Calibration)
# -----------------------------------------------------------------------------
# Designed to restore "Berita Murni" recall by adding mandatory news 
# safeguards for public topics (Sports, Politics, Disaster).

ADVANCED_8B_GOLD_TEMPLATE = """Tugas: Klasifikasikan artikel di bawah sebagai "native ads" atau "berita murni".

[NEWS SAFEGUARD - HARUS BERITA MURNI (KECUALI PROMOSI PRODUK EKSPLISIT)]:
- TOPIK UMUM: Politik Nasional, Ekonomi Global, Tragedi/Kematian, Bencana Alam.
- OLAHRAGA/HIBURAN: Transfer pemain (Haaland), skor pertandingan, info konser.
- LAYANAN PUBLIK: Info Transjakarta, KAI, NASA, atau instansi Pemerintah.
- ATURAN: Jika isi teks hanya "informasi operasional" tanpa ajakan membeli, pilih "berita murni".

[STEALTH CHECKMARK (KHUSUS KOMERSIAL)]:
1. PR NEWSWIRE: Hanya jika ada (GLOBE NEWSWIRE), (PRNewswire) dan memuji satu BRAND.
2. STOCK TICKER: Simbol saham (TSX: ERO, Nasdaq) = INVESTOR RELATION/NATIVE ADS.
3. CORPORATE PRIDE: Hanya jika artikel murni memuji inovasi/penghargaan satu "Perusahaan Swasta" tanpa kritik atau pembanding.

{context}

Judul: {title}
Konten: {content}

Output (JSON):
{{"label": "native ads/berita murni", "alasan": "Penjelasan singkat (Indonesia)"}}"""
