from huggingface_hub import hf_hub_download
import shutil, json

# Download file mentah langsung, tanpa lewat AutoTokenizer/LlamaTokenizer sama sekali
clean_path = hf_hub_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    filename="tokenizer.json"
)

with open(clean_path, 'r', encoding='utf-8') as f:
    tok_json = json.load(f)

print("pre_tokenizer type dari file BERSIH:", tok_json['pre_tokenizer'].get('type'))

# Kalau benar BUKAN Metaspace, timpa file lokal yang rusak
target = '../models/deepseek-r1-llama-8b-native-ads/tokenizer.json'
shutil.copy(clean_path, target)
print("Sudah disalin ke:", target)
