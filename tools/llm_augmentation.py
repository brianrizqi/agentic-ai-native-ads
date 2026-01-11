"""
LLM Data Augmentation
Menggunakan LLM gratis untuk meng-augment dataset dengan generate:
- Pertanyaan yang bervariasi
- Penjelasan yang lebih detail
- Reasoning chains
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict
import time


class LLMAugmenter:
    """
    Augment dataset menggunakan LLM gratis (Groq, Ollama, HuggingFace).
    """
    
    def __init__(self, llm_provider: str = 'groq'):
        """
        Initialize augmenter.
        
        Args:
            llm_provider: 'groq', 'ollama', 'openrouter'
            api_key: API key for groq or openrouter (optional)
        """
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.client = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client berdasarkan provider."""
        
        if self.llm_provider == 'groq':
            try:
                from groq import Groq
                api_key = self.api_key or os.getenv('GROQ_API_KEY')
                if not api_key:
                    print("❌ Groq API key not provided")
                    return None
                client = Groq(api_key=api_key)
                print("✓ Groq client initialized")
                return client
            except ImportError:
                print("Install groq: pip install groq")
                return None
        
        
        elif self.llm_provider == 'huggingface':
            try:
                from transformers import pipeline
                import torch
                device = 0 if torch.cuda.is_available() else -1
                print(f"Loading HuggingFace model (Device: {'GPU' if device==0 else 'CPU'})...")
                # Updated to use GPT-OSS 20B as requested
                generator = pipeline("text-generation", model="openai/gpt-oss-20b", device=device)
                print("✓ HuggingFace client initialized with openai/gpt-oss-20b")
                return generator
            except ImportError:
                print("Install transformers: pip install transformers torch accelerate")
                return None
            except Exception as e:
                print(f"Error loading HF model: {e}")
                return None
        
        return None
        
        return None
    
    def generate_questions(self, content: str, label: str, num_questions: int = 3) -> List[str]:
        """
        Generate variasi pertanyaan untuk satu sample.
        
        Args:
            content: Konten artikel
            label: Label (native_ads, editorial, dll)
            num_questions: Jumlah pertanyaan yang di-generate
        
        Returns:
            List of questions
        """
        
        prompt = f"""Generate {num_questions} different questions in Indonesian that could be asked about this article to determine if it's native advertising or editorial content.

Article excerpt: {content[:300]}
Label: {label}

Generate questions that vary in:
- Directness (direct vs indirect)
- Specificity (general vs specific)
- Focus (content analysis, intent detection, classification)

Return only the questions, one per line."""
        
        if self.llm_provider == 'groq':
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",  # Gratis dan bagus
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=500
            )
            questions_text = response.choices[0].message.content
        
        elif self.llm_provider == 'ollama':
            response = self.client.generate(
                model="llama2",  # atau "mistral"
                prompt=prompt
            )
            questions_text = response['response']
        
        else:
            # Fallback: template-based
            questions_text = self._generate_template_questions(label)
        
        # Parse questions
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and '?' in q]
        return questions[:num_questions]
    
    def generate_explanation(self, content: str, label: str) -> str:
        """
        Generate penjelasan detail mengapa konten diklasifikasikan sebagai label tertentu.
        """
        
        prompt = f"""Analyze this Indonesian news article and provide detailed classification reasoning in JSON format.

**ARTICLE:**
{content}

**GROUND TRUTH LABEL:** {label}

**TASK:**
Analyze based on 4 Native Ads characteristics:
1. **Tone**: Positive/neutral (no criticism) vs Negative/critical
2. **Persuasive Language**: Words that convince/persuade readers
3. **Brand Promotion**: Promotes product/brand/institution
4. **Perspective**: One-sided vs Objective/balanced

**OUTPUT FORMAT (JSON):**
{{
  "label": "{label}",
  "confidence": 0.XX,
  "reasoning": "Detailed analysis in Indonesian covering:
    1. Tone analysis with examples
    2. Persuasive words found (quote specific phrases)
    3. Brand/product mentions and how they're presented
    4. Perspective analysis (one-sided or balanced?)
    5. Conclusion with strongest indicators"
}}

**IMPORTANT:**
- Reasoning must be in Bahasa Indonesia
- Include specific quotes/evidence from article
- Be detailed (200-300 words)
- Confidence: 0.7-0.95 based on clarity of indicators

Output ONLY valid JSON, no additional text:"""
        
        if self.llm_provider == 'groq':
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=800
            )
            explanation = response.choices[0].message.content
        
        elif self.llm_provider == 'ollama':
            response = self.client.generate(
                model="llama2",
                prompt=prompt
            )
            explanation = response['response']
        
        elif self.llm_provider == 'openrouter':
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",  # Fast and cheap
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            explanation = response.choices[0].message.content
            
            # Parse JSON response
            try:
                import json
                import re
                # Extract JSON
                json_match = re.search(r'\{[^{}]*"label"[^{}]*\}', explanation, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                else:
                    return {
                        "label": label,
                        "confidence": 0.75,
                        "reasoning": explanation[:500]
                    }
            except:
                return {
                    "label": label,
                    "confidence": 0.75,
                    "reasoning": explanation[:500]
                }
        
        else:
            explanation = self._generate_template_explanation(content, label)
        
        return explanation
    
    def _generate_template_questions(self, label: str) -> str:
        """Fallback: generate questions using templates."""
        templates = [
            f"Apakah artikel ini termasuk {label}?",
            f"Bagaimana cara mengidentifikasi artikel ini sebagai {label}?",
            f"Apa indikator yang menunjukkan artikel ini adalah {label}?"
        ]
        return '\n'.join(templates)
    
    def _generate_template_explanation(self, content: str, label: str) -> str:
        """Fallback: generate explanation using templates."""
        return f"Artikel ini diklasifikasikan sebagai {label} berdasarkan analisis konten dan karakteristik yang ditemukan."
    
    def augment_dataset(self, input_file: str, output_file: str,
                       content_col: str = 'content',
                       label_col: str = 'label'):
        """
        Augment seluruh dataset dengan LLM-generated content.
        """
        
        print(f"Loading dataset: {input_file}")
        df = pd.read_excel(input_file)
        
        augmented_data = []
        
        print(f"\nAugmenting {len(df)} samples...")
        
        for idx, row in df.iterrows():
            content = str(row[content_col])
            label = str(row[label_col])
            
            print(f"[{idx+1}/{len(df)}] Processing sample {idx}...")
            
            # Generate questions
            questions = self.generate_questions(content, label, num_questions=3)
            
            # Generate explanation
            explanation = self.generate_explanation(content, label)
            
            # Create augmented samples
            for i, question in enumerate(questions):
                augmented_data.append({
                    'id': f'sample_{idx}_q{i}',
                    'original_id': idx,
                    'question': question,
                    'context': content[:1000],
                    'answer': explanation,
                    'label': label
                })
            
            # Rate limiting
            time.sleep(0.5)
        
        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Augmented dataset saved: {output_file}")
        print(f"Original samples: {len(df)}")
        print(f"Augmented samples: {len(augmented_data)}")


def main():
    """Main function."""
    
    print("="*80)
    print("LLM DATA AUGMENTATION")
    print("Generate variasi pertanyaan dan penjelasan menggunakan LLM gratis")
    print("="*80 + "\n")
    
    print("Pilih LLM Provider:")
    print("1. Groq (gratis, cepat, perlu API key)")
    print("2. Ollama (local, gratis, perlu install)")
    print("3. OpenRouter (GPT-4o-mini, perlu API key, ~$5-10 untuk 12k samples)")
    print("4. Skip (gunakan template saja)")
    
    choice = input("\nPilihan (1/2/3/4): ").strip()
    
    provider_map = {
        '1': 'groq',
        '2': 'ollama',
        '3': 'openrouter',
        '4': 'template'
    }
    
    provider = provider_map.get(choice, 'template')
    
    if provider == 'template':
        print("\nMenggunakan template-based generation (tanpa LLM)")
        print("Untuk hasil lebih baik, gunakan Groq atau OpenRouter")
        return
    
    # Get API key if needed
    api_key = None
    if provider in ['groq', 'openrouter']:
        api_key = input(f"\nMasukkan {provider.upper()} API key: ").strip()
        if not api_key:
            print("❌ API key required!")
            return
    
    # Initialize augmenter
    augmenter = LLMAugmenter(llm_provider=provider, api_key=api_key)
    
    if not augmenter.client:
        print("\n❌ Gagal initialize LLM client")
        return
    
    # Augment dataset
    input_file = "native_ads_dataset.xlsx"
    output_file = "data/llm_dataset_augmented.json"
    
    content_col = input("\nNama kolom konten (default: 'content'): ").strip() or 'content'
    label_col = input("Nama kolom label (default: 'label'): ").strip() or 'label'
    
    augmenter.augment_dataset(input_file, output_file, content_col, label_col)
    
    print("\n✓ Augmentation complete!")


if __name__ == "__main__":
    main()
