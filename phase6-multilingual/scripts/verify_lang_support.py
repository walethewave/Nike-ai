"""
Checks every language in lang_config.py against Omnilingual ASR's actual
supported-language list. Requires `pip install omnilingual-asr` (not done
in this repo's base install — see phase2-asr/requirements.txt for why it's
commented out by default: multi-GB, only needed when you're actually
benchmarking).

Usage:
    python3 scripts/verify_lang_support.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lang_config import LANGUAGES  # noqa: E402


def main():
    try:
        from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
    except ImportError:
        print("omnilingual-asr not installed. Run: pip install omnilingual-asr")
        sys.exit(1)

    print(f"{len(supported_langs)} languages supported by Omnilingual ASR.\n")
    all_ok = True
    for key, cfg in LANGUAGES.items():
        actually_supported = cfg.omnilingual_code in supported_langs
        status = "OK" if actually_supported else "MISSING"
        if not actually_supported:
            all_ok = False
        print(f"[{status}] {cfg.display_name} ({cfg.omnilingual_code}) — "
              f"config says verified={cfg.verified}, live check={actually_supported}")
        if actually_supported != cfg.verified:
            print(f"    ^ lang_config.py's `verified` flag is STALE for {key} — update it")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
