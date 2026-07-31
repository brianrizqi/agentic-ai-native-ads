import json

path = '../models/deepseek-r1-llama-8b-native-ads/tokenizer.json'
with open(path, 'r', encoding='utf-8') as f:
    tok_json = json.load(f)

print("Top-level keys:", list(tok_json.keys()))
print()
print("pre_tokenizer section:")
print(json.dumps(tok_json.get('pre_tokenizer'), indent=2, ensure_ascii=False))
print()
print("normalizer section:")
print(json.dumps(tok_json.get('normalizer'), indent=2, ensure_ascii=False))
print()
print("decoder section:")
print(json.dumps(tok_json.get('decoder'), indent=2, ensure_ascii=False)[:1500])
