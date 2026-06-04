#!/usr/bin/env python3
"""Print MagpieTTS pronunciation-control G2P text for one English sentence."""

import argparse
from pathlib import Path

from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p


REPO_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Partially phonemize English text with MagpieTTS IpaG2p.")
    parser.add_argument("text", nargs="+", help="English sentence to phonemize.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for phoneme_probability sampling.")
    args = parser.parse_args()

    g2p = IpaG2p(
        phoneme_dict=REPO_ROOT / "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt",
        heteronyms=REPO_ROOT / "scripts/tts_dataset_files/heteronyms-052722",
        phoneme_probability=0.3,
        ignore_ambiguous_words=False,
        use_chars=True,
        use_stresses=True,
        grapheme_case="mixed",
    )
    if args.seed is not None:
        g2p._rng.seed(args.seed)

    g2p_text = g2p(" ".join(args.text))
    print("".join(g2p_text) if isinstance(g2p_text, list) else str(g2p_text))


if __name__ == "__main__":
    main()
