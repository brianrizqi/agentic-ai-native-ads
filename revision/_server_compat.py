"""Compiler/CUDA workarounds copied verbatim from langchain-refactor/finetune.py (lines 6-119).

Import this BEFORE unsloth on the supercomputer, where no system C compiler is available
and triton has to build through ziglang.
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import sys

# Fix for Triton compiler error: monkey-patch triton to use ziglang if no system compiler
import shutil, sysconfig as _sc, subprocess as _sp

# Flags that zig cc doesn't support (hardcoded by triton in build.py)
_ZIG_BAD_FLAGS = {"-Wno-psabi", "-Wno-format-truncation",
                  "-mno-avx512f", "-fno-semantic-interposition"}

def _find_cc():
    """Find first available C compiler, trying ziglang as last resort."""
    for _c in ["gcc", "clang", "cc", "g++",
               "/usr/bin/gcc", "/usr/bin/clang",
               _sc.get_config_var("CC")]:
        if _c and shutil.which(str(_c).split()[0]):
            return str(_c).split()[0]
    # Try ziglang (pip install ziglang)
    try:
        import ziglang as _zig
        _zig_bin = os.path.join(os.path.dirname(_zig.__file__), "zig")
        if os.path.exists(_zig_bin):
            _wrapper = "/tmp/_zig_cc"
            with open(_wrapper, "w") as _f:
                _f.write(f'#!/bin/sh\nexec "{_zig_bin}" cc "$@"\n')
            os.chmod(_wrapper, 0o755)
            print(f"✅ Triton will use ziglang C compiler via {_wrapper}")
            return _wrapper
    except ImportError:
        pass
    return None

# torch.compile builds Triton kernels in a subprocess pool by default. Those worker
# processes never import this module, so the zig/Python.h patch below does not reach
# them and the build fails. One compile thread keeps compilation in this process.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

_CC = _find_cc()
if _CC:
    os.environ["CC"] = _CC

# Fix CUDA memory fragmentation (recommended when OOM occurs)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Patch subprocess.check_call to fix zig cc incompatibilities:
# 1. Filter GCC-only flags (-Wno-psabi etc. hardcoded by triton)
# 2. Fix missing Python.h by finding it in alternative paths
_orig_check_call = _sp.check_call

def _find_python_h_dir():
    """Return the directory containing Python.h, searching known locations ONLY (no filesystem scan)."""
    import sys as _sys
    pyver = f"python{_sys.version_info.major}.{_sys.version_info.minor}"
    candidates = []
    # 1. venv paths (check VIRTUAL_ENV env var)
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidates += [
            os.path.join(venv, "include", pyver),
            os.path.join(venv, "include"),
            # TensorFlow bundles Python headers (works when python3.x-dev not installed)
            os.path.join(venv, "lib", pyver, "site-packages", "tensorflow",
                         "include", "external", "local_config_python", "python_include"),
        ]
    # 2. Conda env
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        candidates += [os.path.join(conda, "include", pyver), os.path.join(conda, "include")]
    # 3. System paths (exact, no glob)
    candidates += [
        f"/usr/include/{pyver}",
        f"/usr/local/include/{pyver}",
        "/usr/include/python3.12",   # fallback: adjacent version, compatible for triton
        "/usr/include/python3.11",
        "/usr/include/python3.13",
        # Known path from `find / -name Python.h` on this server
        "/home/riset/.local/share/uv/python/cpython-3.13.12-linux-x86_64-gnu/include/python3.13",
    ]
    for d in candidates:
        if d and os.path.exists(os.path.join(d, "Python.h")):
            return d
    return None

_PYTHON_H_DIR = _find_python_h_dir()
if _PYTHON_H_DIR:
    print(f"✅ Found Python.h at: {_PYTHON_H_DIR}")
else:
    print("⚠️  Python.h not found — triton compilation may fail.")

import re as _re
_PYTHON_INC_RE = _re.compile(r"-I.*[/\\]python3\.\d+/?$")  # matches -I.../python3.10 style paths only

def _filtered_check_call(cmd, *args, **kwargs):
    if isinstance(cmd, list) and _CC and "zig" in _CC:
        new_cmd = []
        _py_replaced = False
        for x in cmd:
            if x in _ZIG_BAD_FLAGS:
                continue  # strip incompatible GCC flags
            # Only replace actual Python header include dirs (ends with /python3.x)
            if _PYTHON_H_DIR and _PYTHON_INC_RE.match(x):
                if not _py_replaced:
                    new_cmd.append(f"-I{_PYTHON_H_DIR}")
                    _py_replaced = True
                # skip the bad path
            else:
                new_cmd.append(x)
        # Ensure Python.h dir is in the command
        if _PYTHON_H_DIR and f"-I{_PYTHON_H_DIR}" not in new_cmd:
            new_cmd.append(f"-I{_PYTHON_H_DIR}")
        cmd = new_cmd
    return _orig_check_call(cmd, *args, **kwargs)

_sp.check_call = _filtered_check_call
print(f"✅ Patched subprocess.check_call (CC={_CC or 'not found'}, Python.h={_PYTHON_H_DIR or 'not found'})")
