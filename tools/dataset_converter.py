"""
Dataset Converter: Deep Learning Format → LLM Format
Mengkonversi dataset native ads (konten + label) menjadi format QnA untuk LLM
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import random


class DatasetConverter:
    """
    Converter untuk mengubah dataset Deep Learning (konten + label)
    menjadi format LLM (QnA, Instruction-Response, Chat)
    """
    
    def __init__(self, input_file: str):
        """
        Initialize converter.
        
        Args:
            input_file: Path ke file Excel dataset (native_ads_dataset.xlsx)
        """
        self.input_file = input_file
        self.df = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load dataset dari Excel."""
        print(f"Loading dataset from: {self.input_file}")
        # Read from 'Clean' sheet to get all 12088 rows
        self.df = pd.read_excel(self.input_file, sheet_name='Clean')
        print(f"✓ Loaded {len(self.df)} samples")
        print(f"Columns: {list(self.df.columns)}")
        print(f"\nSample data:")
        print(self.df.head())
    
    def convert_to_qna_format(self, 
                              content_column: str = 'content',
                              label_column: str = 'label') -> List[Dict[str, str]]:
        """
        Konversi ke format QnA (Question-Answer).
        
        Format:
        {
            "question": "Apakah artikel berikut ini termasuk native advertising?",
            "context": "[konten artikel]",
            "answer": "Ya/Tidak, ini adalah [label]. Alasannya: ..."
        }
        """
        print("\nConverting to QnA format...")
        
        qna_data = []
        
        # Template pertanyaan
        question_templates = [
            "Apakah artikel berikut ini termasuk native advertising?",
            "Klasifikasikan konten berikut: apakah ini native ads atau konten editorial?",
            "Analisis artikel ini dan tentukan apakah ini native advertising.",
            "Identifikasi jenis konten berikut: native ads, editorial, atau sponsored content?",
            "Berdasarkan konten berikut, apakah ini termasuk iklan terselubung (native ads)?"
        ]
        
        for idx, row in self.df.iterrows():
            content = str(row[content_column])
            label = str(row[label_column])
            
            # Pilih random question template
            question = random.choice(question_templates)
            
            # Generate answer berdasarkan label
            answer = self._generate_answer(content, label)
            
            qna_data.append({
                "id": f"sample_{idx}",
                "question": question,
                "context": content[:1000],  # Limit panjang konten
                "answer": answer,
                "label": label
            })
        
        print(f"✓ Converted {len(qna_data)} samples to QnA format")
        return qna_data
    
    def convert_to_instruction_format(self,
                                     content_column: str = 'content',
                                     label_column: str = 'label') -> List[Dict[str, str]]:
        """
        Konversi ke format Instruction-Response (untuk fine-tuning).
        
        Format:
        {
            "instruction": "Klasifikasikan artikel berikut sebagai native ads atau editorial content.",
            "input": "[konten artikel]",
            "output": "[label] - [penjelasan]"
        }
        """
        print("\nConverting to Instruction-Response format...")
        
        instruction_data = []
        
        # Template instruction
        instruction_templates = [
            "Klasifikasikan artikel berikut sebagai native advertising atau konten editorial.",
            "Analisis konten berikut dan tentukan apakah ini native ads.",
            "Identifikasi jenis konten: native advertising, editorial content, atau sponsored content.",
            "Deteksi apakah artikel berikut mengandung native advertising.",
            "Evaluasi konten dan klasifikasikan sebagai native ads atau bukan."
        ]
        
        for idx, row in self.df.iterrows():
            content = str(row[content_column])
            label = str(row[label_column])
            
            instruction = random.choice(instruction_templates)
            output = self._generate_detailed_output(content, label)
            
            instruction_data.append({
                "id": f"sample_{idx}",
                "instruction": instruction,
                "input": content[:1000],
                "output": output
            })
        
        print(f"✓ Converted {len(instruction_data)} samples to Instruction format")
        return instruction_data
    
    def convert_to_chat_format(self,
                               content_column: str = 'content',
                               label_column: str = 'label') -> List[Dict[str, Any]]:
        """
        Konversi ke format Chat (untuk ChatGPT, Claude, dll).
        
        Format:
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
        """
        print("\nConverting to Chat format...")
        
        chat_data = []
        
        system_message = """Anda adalah expert dalam mendeteksi native advertising pada portal berita elektronik. 
Tugas Anda adalah menganalisis artikel dan mengklasifikasikan apakah artikel tersebut termasuk:
- Native Advertising: Iklan yang menyerupai konten editorial
- Editorial Content: Konten berita murni
- Sponsored Content: Konten bersponsor dengan label jelas
- Advertorial: Iklan dalam format editorial

Berikan klasifikasi dan penjelasan yang detail."""
        
        for idx, row in self.df.iterrows():
            content = str(row[content_column])
            label = str(row[label_column])
            
            user_message = f"Klasifikasikan artikel berikut:\n\n{content[:800]}"
            assistant_message = self._generate_detailed_output(content, label)
            
            chat_data.append({
                "id": f"sample_{idx}",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message}
                ]
            })
        
        print(f"✓ Converted {len(chat_data)} samples to Chat format")
        return chat_data
    
    def _generate_answer(self, content: str, label: str) -> str:
        """Generate answer untuk QnA format."""
        
        # Mapping label ke penjelasan
        label_explanations = {
            'native_ads': 'Ya, ini adalah Native Advertising. Konten ini dirancang menyerupai artikel editorial namun memiliki tujuan promosi.',
            'editorial': 'Tidak, ini adalah konten editorial murni. Artikel ini fokus pada informasi faktual tanpa agenda promosi.',
            'sponsored': 'Ini adalah Sponsored Content. Konten bersponsor dengan label yang jelas.',
            'advertorial': 'Ya, ini adalah Advertorial. Iklan dalam format editorial dengan tujuan promosi produk/layanan.'
        }
        
        # Normalize label
        label_lower = label.lower().replace(' ', '_')
        
        # Get explanation atau default
        explanation = label_explanations.get(label_lower, f'Ini termasuk kategori: {label}')
        
        return explanation
    
    def _generate_detailed_output(self, content: str, label: str) -> str:
        """Generate detailed output untuk Instruction/Chat format."""
        
        # Analisis sederhana konten
        content_lower = content.lower()
        
        # Deteksi promotional words
        promo_words = ['terbaik', 'solusi', 'rekomendasi', 'wajib', 'harus', 'promo', 'diskon']
        promo_count = sum(1 for word in promo_words if word in content_lower)
        
        # Deteksi brand mentions (simplified)
        has_brand = any(word in content_lower for word in ['produk', 'layanan', 'brand', 'merek'])
        
        # Generate output
        output = f"**Klasifikasi**: {label}\n\n"
        output += f"**Analisis**:\n"
        
        if label.lower() in ['native_ads', 'native advertising']:
            output += "- Konten terdeteksi sebagai Native Advertising\n"
            output += f"- Ditemukan {promo_count} kata promosi\n"
            if has_brand:
                output += "- Terdapat brand mentions dalam konten\n"
            output += "- Struktur menyerupai artikel editorial namun memiliki tujuan komersial\n"
        elif label.lower() in ['editorial', 'editorial content']:
            output += "- Konten editorial murni\n"
            output += "- Fokus pada informasi faktual\n"
            output += "- Tidak ada agenda promosi yang terdeteksi\n"
        else:
            output += f"- Konten termasuk kategori: {label}\n"
        
        output += f"\n**Confidence**: High"
        
        return output
    
    def save_to_json(self, data: List[Dict], output_file: str):
        """Save converted data ke JSON."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to: {output_file}")
    
    def save_to_jsonl(self, data: List[Dict], output_file: str):
        """Save converted data ke JSONL (untuk fine-tuning)."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved to: {output_file}")


def main():
    """Main function untuk konversi dataset."""
    
    print("="*80)
    print("DATASET CONVERTER: Deep Learning → LLM Format")
    print("Native Ads Detection - Postdoc Research")
    print("="*80 + "\n")
    
    # Path ke dataset
    input_file = "native_ads_dataset.xlsx"
    
    # Check if file exists
    if not Path(input_file).exists():
        print(f"❌ File tidak ditemukan: {input_file}")
        print("\nPastikan file native_ads_dataset.xlsx ada di direktori yang sama.")
        return
    
    # Initialize converter
    converter = DatasetConverter(input_file)
    
    # Tanyakan nama kolom
    print("\n" + "="*80)
    print("KONFIGURASI KOLOM")
    print("="*80)
    
    content_col = input("Nama kolom untuk konten artikel (default: 'content'): ").strip() or 'content'
    label_col = input("Nama kolom untuk label (default: 'label'): ").strip() or 'label'
    
    # Convert ke berbagai format
    print("\n" + "="*80)
    print("KONVERSI DATASET")
    print("="*80)
    
    # 1. QnA Format
    qna_data = converter.convert_to_qna_format(content_col, label_col)
    converter.save_to_json(qna_data, 'data/llm_dataset_qna.json')
    
    # 2. Instruction Format
    instruction_data = converter.convert_to_instruction_format(content_col, label_col)
    converter.save_to_json(instruction_data, 'data/llm_dataset_instruction.json')
    converter.save_to_jsonl(instruction_data, 'data/llm_dataset_instruction.jsonl')
    
    # 3. Chat Format (OpenAI fine-tuning format)
    chat_data = converter.convert_to_chat_format(content_col, label_col)
    converter.save_to_jsonl(chat_data, 'data/llm_dataset_chat.jsonl')
    
    # Summary
    print("\n" + "="*80)
    print("KONVERSI SELESAI")
    print("="*80)
    print(f"\nTotal samples: {len(converter.df)}")
    print(f"\nOutput files:")
    print(f"  1. data/llm_dataset_qna.json - QnA format")
    print(f"  2. data/llm_dataset_instruction.json - Instruction format (JSON)")
    print(f"  3. data/llm_dataset_instruction.jsonl - Instruction format (JSONL)")
    print(f"  4. data/llm_dataset_chat.jsonl - Chat format (OpenAI)")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. QnA Format: Untuk RAG, question-answering systems
2. Instruction Format: Untuk fine-tuning LLM (Llama, Mistral, dll)
3. Chat Format: Untuk fine-tuning ChatGPT/GPT-4

Untuk fine-tuning gratis, gunakan:
- Groq (gratis, cepat)
- Ollama (local, gratis)
- HuggingFace (gratis dengan GPU)
- Google Colab (gratis dengan GPU)
    """)


if __name__ == "__main__":
    main()
