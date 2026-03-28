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

Judul: {title}
Konten: {content}

{context}

Analisis:
"""

simple_local_prompt = PromptTemplate(
    input_variables=["title", "content", "context"],
    template=SIMPLE_LOCAL_PROMPT_TEMPLATE
)


# Condensed version for small models (< 3B) to avoid instruction fade
CONDENSED_TRAINING_PROMPT_TEMPLATE = """{context}
TUGAS: Klasifikasikan berita di bawah sebagai "native ads" atau "berita murni".

Judul: {title}
Konten: {content}

Output (JSON):
{{"label": """

# Training-specific prompt template (matches inference format exactly)
TRAINING_PROMPT_TEMPLATE = """Klasifikasikan berita berikut sebagai "native ads" atau "berita murni".
- Berita objektif (seperti sejarah, kriminal, atau penemuan ilmiah) adalah "berita murni".
- Jika ragu atau konten bersifat informatif belaka, pilih "berita murni".

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
{{"label": "native ads" atau "berita murni", "confidence": 0.0-1.0, "reasoning": "alasan singkat (max 150 karakter)"}}

Klasifikasi:
"""

training_prompt = PromptTemplate(
    input_variables=["title", "content", "context"],
    template=TRAINING_PROMPT_TEMPLATE
)
