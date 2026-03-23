"""
Create Irrelevant Vector Store
==============================
Membuat vectorstore FAISS berisi dokumen-dokumen yang tidak relevan 
dengan native ads (berita olahraga, sains, hewan, dll).
Digunakan sebagai baseline 'RAG Bukan Data Kita' dalam studi ablasi.
"""

import os
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IRRELEVANT_DATA = [
    # Sains & Astronomi
    "Lubang hitam adalah wilayah ruang-waktu yang tarikan gravitasinya sangat kuat sehingga tidak ada materi atau radiasi yang dapat meloloskan diri.",
    "Teleskop James Webb menangkap gambar galaksi jauh dengan detail yang belum pernah terlihat sebelumnya, mengungkap rahasia alam semesta awal.",
    "Mars memiliki gunung berapi terbesar di tata surya, Olympus Mons, yang tingginya hampir tiga kali lipat Everest.",
    
    # Olahraga
    "Pertandingan final liga champions akan diadakan di stadion Wembley, mempertemukan dua raksasa sepak bola Eropa.",
    "Atlet lari maraton Kenya memecahkan rekor dunia baru di Berlin dengan catatan waktu di bawah dua jam dua menit.",
    "Bulu tangkis merupakan olahraga yang sangat populer di Indonesia, dengan banyak prestasi di tingkat olimpiade.",
    
    # Alam & Hewan
    "Hutan hujan Amazon menghasilkan sekitar 20% oksigen dunia dan menjadi rumah bagi jutaan spesies tumbuhan dan hewan.",
    "Panda raksasa menghabiskan sekitar 10 hingga 16 jam sehari untuk makan bambu guna memenuhi kebutuhan energinya.",
    "Terumbu karang sering disebut sebagai hutan hujan laut karena keanekaragaman hayati yang sangat tinggi di dalamnya.",
    
    # Memasak & Kuliner
    "Rendang adalah masakan daging bercita rasa pedas yang menggunakan campuran berbagai bumbu dan rempah-rempah asli Indonesia.",
    "Teknik memasak sous-vide melibatkan vakum makanan dalam kantong plastik dan merendamnya dalam air pada suhu yang sangat terkontrol.",
    "Kopi Arabika dikenal memiliki profil rasa yang lebih kompleks dan tingkat keasaman yang lebih tinggi dibanding Robusta.",
    
    # Sejarah & Budaya
    "Candi Borobudur dibangun pada abad ke-9 dan merupakan monumen Buddha terbesar di dunia yang terletak di Magelang.",
    "Revolusi Industri dimulai di Britania Raya pada abad ke-18 dan mengubah cara manusia memproduksi barang secara massal.",
    "Batik telah diakui oleh UNESCO sebagai Warisan Kemanusiaan untuk Budaya Lisan dan Nonbendawi sejak tahun 2009.",
    
    # Kesehatan (Umum)
    "Mengatur pola tidur yang teratur sangat penting untuk menjaga kesehatan mental dan fungsi kognitif otak sehari-hari.",
    "Paparan sinar matahari pagi membantu tubuh memproduksi vitamin D yang penting untuk kesehatan tulang dan sistem imun.",
    "Minum air putih yang cukup membantu metabolisme tubuh dan menjaga kelembapan kulit secara alami.",
]

def create_irrelevant_db(output_dir: str, embedding_model: str):
    logger.info(f"🚀 Creating irrelevant vectorstore in: {output_dir}")
    
    # Pre-checks
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize embeddings
    logger.info(f"Loading embedding model: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Create documents
    documents = []
    for i, content in enumerate(IRRELEVANT_DATA):
        doc = Document(
            page_content=content,
            metadata={
                "id": f"irr_{i}",
                "label": "irrelevant",
                "source": "external_fake_news"
            }
        )
        documents.append(doc)
    
    logger.info(f"Creating FAISS index with {len(documents)} snippets...")
    
    # Create and save FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(output_dir)
    
    logger.info(f"✅ Irrelevant vectorstore saved to: {output_dir}")
    logger.info(f"Total documents: {len(documents)}")

if __name__ == "__main__":
    # Default config sesuai dengan setup_vectorstore.py
    OUTPUT_DIR = "data/vectorstore_irrelevant"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    create_irrelevant_db(OUTPUT_DIR, EMBEDDING_MODEL)
