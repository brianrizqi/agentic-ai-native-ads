"""
patch_tokenizer_loading.py

Patches agents/classification_agent.py so that local models are loaded with
a safe tokenizer-loading path (tokenizers.Tokenizer.from_file +
PreTrainedTokenizerFast) instead of AutoTokenizer.from_pretrained().

This works around a confirmed transformers v5 bug (huggingface/transformers
issue #45488): LlamaTokenizer.__init__ unconditionally overwrites a correct
ByteLevel BPE pre-tokenizer with a broken Metaspace one for DeepSeek/Llama-3
style tokenizers, silently deleting all spaces from decoded text.

Usage (run from the langchain-refactor/ directory):
    python patch_tokenizer_loading.py

It edits agents/classification_agent.py in place and prints a diff-like
summary. A backup is written to agents/classification_agent.py.bak first.
"""

import shutil
import sys
from pathlib import Path

TARGET = Path("agents/classification_agent.py")

OLD_BLOCK = '''                # Phase 65: Nuclear Stability Reloaded (Safe Tokenizer Loading)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, token=hf_token, trust_remote_code=True, 
                        fix_mistral_regex=True
                    )
                except TypeError:
                    # Fallback for transformers versions where this is already handled or causes conflicts
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, token=hf_token, trust_remote_code=True
                    )'''

NEW_BLOCK = '''                # Phase 65+: Safe Tokenizer Loading
                # Bypasses a confirmed transformers v5 bug (huggingface/transformers
                # issue #45488): AutoTokenizer/LlamaTokenizer silently rewrites a
                # correct ByteLevel BPE pre-tokenizer into a broken Metaspace one for
                # DeepSeek/Llama-3 style tokenizers, deleting all spaces on decode.
                # We load tokenizer.json directly via the Rust `tokenizers` binding
                # and wrap it in a generic PreTrainedTokenizerFast, which does not
                # go through the buggy LlamaTokenizer.__init__ path at all.
                tokenizer = None
                local_tokenizer_json = os.path.join(self.model_name, "tokenizer.json") \\
                    if os.path.isdir(self.model_name) else None

                if local_tokenizer_json and os.path.isfile(local_tokenizer_json):
                    try:
                        from tokenizers import Tokenizer as _RawTokenizer
                        raw_tok = _RawTokenizer.from_file(local_tokenizer_json)
                        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tok)
                        # Carry over special-token config from tokenizer_config.json
                        # if present, so chat templates / added tokens still work.
                        try:
                            tokenizer_ref = AutoTokenizer.from_pretrained(
                                self.model_name, token=hf_token, trust_remote_code=True
                            )
                            tokenizer.bos_token = getattr(tokenizer_ref, "bos_token", tokenizer.bos_token)
                            tokenizer.eos_token = getattr(tokenizer_ref, "eos_token", tokenizer.eos_token)
                            tokenizer.pad_token = getattr(tokenizer_ref, "pad_token", tokenizer.eos_token)
                            if getattr(tokenizer_ref, "chat_template", None):
                                tokenizer.chat_template = tokenizer_ref.chat_template
                        except Exception as _meta_err:
                            print(f"WARNING: could not copy special-token metadata: {_meta_err}")
                        print(f"DEBUG: Loaded SAFE bypass tokenizer for {self.model_name} "
                              f"(raw tokenizer.json, no LlamaTokenizer override)")
                    except Exception as _safe_load_err:
                        print(f"WARNING: safe tokenizer bypass failed ({_safe_load_err}), "
                              f"falling back to AutoTokenizer.")
                        tokenizer = None

                if tokenizer is None:
                    # Fallback: standard AutoTokenizer path (used for Hub repo ids,
                    # or local folders without a tokenizer.json).
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name, token=hf_token, trust_remote_code=True,
                            fix_mistral_regex=True
                        )
                    except TypeError:
                        tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name, token=hf_token, trust_remote_code=True
                        )'''

REQUIRED_IMPORTS = [
    ("import os", "os"),
    ("from transformers import PreTrainedTokenizerFast", "PreTrainedTokenizerFast"),
]


def main():
    if not TARGET.exists():
        print(f"FATAL: {TARGET} not found. Run this script from the langchain-refactor/ directory.")
        sys.exit(1)

    src = TARGET.read_text(encoding="utf-8")

    if OLD_BLOCK not in src:
        if "Safe Tokenizer Loading" in src and "safe tokenizer bypass" in src:
            print("Already patched (safe bypass block found). Nothing to do.")
            sys.exit(0)
        print("FATAL: expected code block not found verbatim in classification_agent.py.")
        print("The file may have changed since this patch was written. Aborting without changes.")
        sys.exit(1)

    backup_path = TARGET.with_suffix(".py.bak")
    shutil.copy(TARGET, backup_path)
    print(f"Backup written to {backup_path}")

    patched = src.replace(OLD_BLOCK, NEW_BLOCK)

    # Ensure `import os` exists near the top.
    if "\nimport os\n" not in patched and not patched.startswith("import os\n"):
        # Insert after the first import line.
        lines = patched.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                lines.insert(i, "import os")
                break
        patched = "\n".join(lines)
        print("Inserted 'import os'.")

    # Ensure PreTrainedTokenizerFast is imported alongside AutoTokenizer.
    patched = patched.replace(
        "from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, GenerationConfig",
        "from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, GenerationConfig, PreTrainedTokenizerFast",
    )

    TARGET.write_text(patched, encoding="utf-8")
    print(f"Patched {TARGET} successfully.")
    print("\nNext steps:")
    print("  1. Re-run the tokenizer decode sanity test for DeepSeek to confirm it still works.")
    print("  2. Re-run evaluate_model.py for DeepSeek with the clean vectorstore/test split.")
    print("  3. Spot-check Gemma/Qwen still load fine (they should be unaffected, "
          "but this patch changes the code path for ALL local models).")


if __name__ == "__main__":
    main()
