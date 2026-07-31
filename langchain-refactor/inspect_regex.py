import json

path = '../models/deepseek-r1-llama-8b-native-ads/tokenizer.json'
with open(path, 'r', encoding='utf-8') as f:
    tok_json = json.load(f)

def find_regex(node, path_str=""):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == 'Regex':
                print(f"Found Regex at {path_str}:")
                print(repr(v))
                print()
            else:
                find_regex(v, path_str + "." + k)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            find_regex(item, path_str + f"[{i}]")

find_regex(tok_json)
