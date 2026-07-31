from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

raw_tok = Tokenizer.from_file('../models/deepseek-r1-llama-8b-native-ads/tokenizer.json')
tok = PreTrainedTokenizerFast(tokenizer_object=raw_tok)

test_text = 'Artikel bersifat umum tanpa unsur promosi atau iklan.'
ids = tok.encode(test_text)
decoded = tok.decode(ids, skip_special_tokens=True)
print('ORIGINAL :', repr(test_text))
print('DECODED  :', repr(decoded))
print('MATCH:', test_text.strip() == decoded.strip())
