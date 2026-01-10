"""
Classification Prompts for Native Ads Detection
Provides prompt templates for LLM-based classification
"""

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


# Few-shot examples for classification
CLASSIFICATION_EXAMPLES = [
    {
        "content": "Promo spesial BRI di HUT ke-128, diskon hingga Rp1.28 juta untuk berbagai produk perbankan.",
        "label": "native ads",
        "reasoning": "Promosi produk bank, diskon, ajakan transaksi."
    },
    {
        "content": "Presiden Jokowi menunjuk Ridwan Kamil sebagai kurator infrastruktur IKN pada hari Selasa.",
        "label": "berita murni",
        "reasoning": "Berita faktual kebijakan pemerintah, tidak ada promosi produk."
    },
    {
        "content": "Dapatkan cashback hingga 50% untuk pembelian produk elektronik di Tokopedia hari ini!",
        "label": "native ads",
        "reasoning": "Ajakan beli dengan cashback, promosi e-commerce."
    },
    {
        "content": "Inflasi Indonesia pada bulan Maret tercatat 4.97%, turun dari bulan sebelumnya menurut BPS.",
        "label": "berita murni",
        "reasoning": "Laporan ekonomi faktual dari sumber resmi, tidak ada ajakan beli."
    },
    {
        "content": "Investasi emas Antam kini lebih mudah dengan aplikasi baru, gratis biaya admin bulan ini.",
        "label": "native ads",
        "reasoning": "Promosi aplikasi investasi, gratis biaya, ajakan menggunakan produk."
    },
    {
        "content": "Mahkamah Agung menolak kasasi terdakwa kasus korupsi dengan vonis 8 tahun penjara.",
        "label": "berita murni",
        "reasoning": "Berita hukum faktual, tidak ada promosi atau ajakan komersial."
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
EXPLANATION_PROMPT_TEMPLATE = """Berikan penjelasan detail tentang klasifikasi berikut:

Konten: {content}
Label: {label}
Confidence: {confidence}
Reasoning: {reasoning}

Jelaskan dalam Bahasa Indonesia:
1. Mengapa konten ini diklasifikasikan sebagai "{label}"?
2. Apa indikator kunci yang mendukung klasifikasi ini?
3. Apakah ada elemen yang membuat klasifikasi ini ambigu?

Penjelasan:"""

explanation_prompt = PromptTemplate(
    input_variables=["content", "label", "confidence", "reasoning"],
    template=EXPLANATION_PROMPT_TEMPLATE
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
