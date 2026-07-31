from tokenizers import Tokenizer

raw_tok = Tokenizer.from_file('../models/deepseek-r1-llama-8b-native-ads/tokenizer.json')

test_text = 'Artikel bersifat umum'
encoding = raw_tok.encode(test_text)

print("Tokens:", encoding.tokens)
print("IDs:", encoding.ids)
print()

# decode manually step by step
decoded = raw_tok.decode(encoding.ids)
print("Decoded (raw tokenizers lib):", repr(decoded))
