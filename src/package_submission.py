"""Build the submission zip.

The zip holds the two output files, the documentation, the README and requirements,
the main pipeline modules listed in PIPELINE, every .py/.sh/.md file under
src/final_steps/ (the post-pipeline steps that produced the final file) and the
word lists the pipeline learns into data/ (translit_map.json from learn_translit.py,
decoy_words.json from learn_decoy_words.py, replacement_words.json from decoy_families.py)
and v10/l2_words.json, the level-2 decoy words read by final_steps/france_rerun/ofeats.py.

Usage: python package_submission.py <team> [matching.tsv] [candidate_pairs.tsv] [--zip out.zip]
"""
import argparse
import zipfile
from pathlib import Path

from config import OUTPUT_DIR, PROJECT_DIR

PIPELINE = [
    "config.py",
    "normalize.py",
    "learn_translit.py",
    "build_normalized.py",
    "block_candidates.py",
    "evaluate_blocking.py",
    "build_features.py",
    "train_matcher.py",
    "learn_decoy_words.py",
    "decoy_families.py",
    "decoy_veto.py",
    "odd_features.py",
    "build_odd_features.py",
    "train_stage2.py",
    "expected_f.py",
    "address_rules.py",
    "france_rules.py",
    "cross_encoder.py",
    "build_ce_pairs.py",
    "train_cross_encoder.py",
    "score_cross_encoder.py",
    "train_stacker.py",
    "rescue_candidates.py",
    "score_rescue.py",
    "train_rescue.py",
    "predict_submission.py",
    "package_submission.py",
]

WORD_LISTS = ["translit_map.json", "decoy_words.json", "replacement_words.json", "v10/l2_words.json"]
FINAL_STEPS_SUFFIXES = {".py", ".sh", ".md"}

SUBMISSION_DIR = PROJECT_DIR / "submission"
CODE_ROOT = "code/business_entity_resolution"


def final_steps_files():
    """Return every .py, .sh and .md file under src/final_steps, sorted."""
    root = PROJECT_DIR / "src" / "final_steps"
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in FINAL_STEPS_SUFFIXES and "__pycache__" not in p.parts
    )


def entries(matching, candidates):
    """Return the (source path, name inside the zip) pairs to package."""
    out = [
        (matching, "output/matching_results.tsv"),
        (candidates, "output/candidate_pairs.tsv"),
        (SUBMISSION_DIR / "README.md", f"{CODE_ROOT}/README.md"),
        (PROJECT_DIR / "requirements.txt", f"{CODE_ROOT}/requirements.txt"),
        (SUBMISSION_DIR / "Documentation_template.md", "Documentation_template.md"),
    ]
    out += [(PROJECT_DIR / "src" / name, f"{CODE_ROOT}/src/{name}") for name in PIPELINE]
    src_root = PROJECT_DIR / "src"
    out += [(p, f"{CODE_ROOT}/src/{p.relative_to(src_root).as_posix()}") for p in final_steps_files()]
    out += [(PROJECT_DIR / "data" / name, f"{CODE_ROOT}/data/{name}") for name in WORD_LISTS]
    return out


def main():
    """Parse the command line and write the submission zip."""
    ap = argparse.ArgumentParser(description="Build the submission zip")
    ap.add_argument("team")
    ap.add_argument("matching", nargs="?", default=str(OUTPUT_DIR / "best" / "matching_results.tsv"))
    ap.add_argument("candidates", nargs="?", default=str(OUTPUT_DIR / "best" / "candidate_pairs.tsv"))
    ap.add_argument("--zip", default=None)
    args = ap.parse_args()
    zip_path = Path(args.zip) if args.zip else OUTPUT_DIR / f"{args.team}_submission.zip"
    files = entries(Path(args.matching), Path(args.candidates))
    missing = [str(src) for src, _ in files if not src.exists()]
    if missing:
        raise SystemExit("missing files:\n" + "\n".join(missing))
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arcname in files:
            zf.write(src, arcname)
            print(f"added {arcname}")
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
