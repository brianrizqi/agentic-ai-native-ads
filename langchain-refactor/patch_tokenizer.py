import json
import shutil

path = '../models/deepseek-r1-llama-8b-native-ads/tokenizer.json'
shutil.copy(path, path + '.bak')  # backup dulu

with open(path, 'r', encoding='utf-8') as f:
    tok_json = json.load(f)

BROKEN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
FIXED = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"

found = False

def patch(node):
    global found
    if isinstance(node, dict):
        for k, v in node.items():
            if k == 'Regex' and v == BROKEN:
                node[k] = FIXED
                found = True
            else:
                patch(v)
    elif isinstance(node, list):
        for item in node:
            patch(item)

patch(tok_json)
print('Pattern found and patched:', found)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(tok_json, f, ensure_ascii=False)
