"""
batch_eval_ml.py
────────────────
Batch evaluation script for MCP_test_v3 — Machine Learning sheet.

Stop anytime with Ctrl+C — re-run to resume from where you left off.
"""

import sys
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
EXCEL_PATH             = r"C:\Users\srima\OneDrive\Desktop\INSY669\Group Project\Question Bank with LLM answers.xlsx"
SHEET_NAME             = "Machine Learning"
ANSWER_COL             = "Answers"
QUESTION_COL           = "Questions"
OUTPUT_CSV             = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "batch_results_Machine_Learning.csv")

BATCH_SIZE             = 10   # rows per batch
SLEEP_BETWEEN_ROWS     = 5    # seconds between rows (rate limiting)
SLEEP_BETWEEN_BATCHES  = 60   # seconds between batches (rate limiting)
MAX_RESULTS_PER_SOURCE = 10
SEARCH_TIMEOUT         = 120  # seconds before giving up on a hung search
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
from MCP_test_v3 import search_by_claims, compute_claim_verdict


def load_done_claims(output_csv):
    if not os.path.exists(output_csv):
        return set()
    try:
        existing = pd.read_csv(output_csv)
        return set(existing["claim_text"].dropna().tolist())
    except Exception:
        return set()


def init_output_csv(output_csv):
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=[
            "row_index", "question", "claim_text",
            "papers_found", "agree", "disagree", "neutral",
            "validity_score", "agreement_rate", "disagreement_rate",
        ]).to_csv(output_csv, index=False)


def append_result(output_csv, row_dict):
    pd.DataFrame([row_dict]).to_csv(output_csv, mode="a", header=False, index=False)


def run_batch_evaluation():
    print(f"Loading dataset — sheet: '{SHEET_NAME}'")
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    print(f"Loaded {len(df)} rows")

    done_claims = load_done_claims(OUTPUT_CSV)
    print(f"Already processed: {len(done_claims)} row(s) — will skip these.")
    print(f"Remaining:         {len(df) - len(done_claims)} row(s)\n")

    init_output_csv(OUTPUT_CSV)

    completed = 0
    skipped   = 0

    try:
        for batch_start in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[batch_start : batch_start + BATCH_SIZE]
            batch_num     = batch_start // BATCH_SIZE + 1
            total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"\n{'#'*70}")
            print(f"# Batch {batch_num} / {total_batches}  "
                  f"(rows {batch_start + 1}–{min(batch_start + BATCH_SIZE, len(df))})")
            print(f"{'#'*70}")

            rows_in_batch = list(batch.iterrows())
            for local_i, (abs_i, row) in enumerate(rows_in_batch):
                claim_text = str(row.get(ANSWER_COL, "")).strip()
                question   = str(row.get(QUESTION_COL, "")).strip()

                if not claim_text or claim_text.lower() == "nan":
                    print(f"\n[Row {abs_i + 1}/{len(df)}] Empty answer — skipping.")
                    skipped += 1
                    continue

                if claim_text in done_claims:
                    print(f"\n[Row {abs_i + 1}/{len(df)}] Already done — skipping.")
                    skipped += 1
                    continue

                print(f"\n{'─'*70}")
                print(f"[Row {abs_i + 1}/{len(df)}]")
                print(f"  Q: {question[:100]}")
                print(f"  A: {claim_text[:150]}")
                print(f"{'─'*70}")

                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            search_by_claims, claim_text, MAX_RESULTS_PER_SOURCE
                        )
                        try:
                            claim_results = future.result(timeout=SEARCH_TIMEOUT)
                        except FuturesTimeout:
                            print(f"\n  TIMEOUT ({SEARCH_TIMEOUT}s) — skipping row, will retry next run.")
                            continue

                    total_agree    = 0
                    total_disagree = 0
                    total_neutral  = 0
                    total_papers   = 0

                    for claim_data in claim_results:
                        verdict = compute_claim_verdict(claim_data)
                        ct = verdict["claim_totals"]
                        total_agree    += ct["agree"]
                        total_disagree += ct["disagree"]
                        total_neutral  += ct["neutral"]
                        total_papers   += len(
                            claim_data.get("results", {}).get("combined", [])
                        )

                    total_sents       = total_agree + total_disagree + total_neutral
                    validity_score    = total_agree / total_sents if total_sents else 0.0
                    disagreement_rate = total_disagree / total_sents if total_sents else 0.0

                    result = {
                        "row_index":          abs_i,
                        "question":           question,
                        "claim_text":         claim_text,
                        "papers_found":       total_papers,
                        "agree":              total_agree,
                        "disagree":           total_disagree,
                        "neutral":            total_neutral,
                        "validity_score":     round(validity_score, 4),
                        "agreement_rate":     round(validity_score, 4),
                        "disagreement_rate":  round(disagreement_rate, 4),
                    }

                    append_result(OUTPUT_CSV, result)
                    done_claims.add(claim_text)
                    completed += 1

                    print(f"\n  Validity score: {validity_score:.1%}  "
                          f"(agree={total_agree}  disagree={total_disagree}  "
                          f"neutral={total_neutral}  papers={total_papers})")
                    print(f"  Saved to {OUTPUT_CSV}")

                except Exception as e:
                    print(f"\n  ERROR on row {abs_i + 1}: {e}")
                    print("  Row will be retried on next run.")

                if local_i < len(rows_in_batch) - 1:
                    print(f"\n  Waiting {SLEEP_BETWEEN_ROWS}s...")
                    time.sleep(SLEEP_BETWEEN_ROWS)

            if batch_start + BATCH_SIZE < len(df):
                print(f"\n--- Batch {batch_num} complete. "
                      f"Waiting {SLEEP_BETWEEN_BATCHES}s before next batch ---")
                time.sleep(SLEEP_BETWEEN_BATCHES)

    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print("Stopped by user (Ctrl+C).")
        print(f"  Completed this session : {completed} row(s)")
        print(f"  Skipped (already done) : {skipped} row(s)")
        print(f"  Results saved to       : {OUTPUT_CSV}")
        print("  Re-run this script to resume from where you left off.")
        print(f"{'='*70}")
        sys.exit(0)

    print(f"\n{'='*70}")
    print("All rows processed!")
    print(f"  Completed : {completed} row(s)")
    print(f"  Skipped   : {skipped} row(s)")
    print(f"  Output    : {OUTPUT_CSV}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing results and start from row 0")
    args = parser.parse_args()

    if args.reset and os.path.exists(OUTPUT_CSV):
        try:
            os.remove(OUTPUT_CSV)
            print(f"Reset: deleted {OUTPUT_CSV}")
        except OSError:
            # OneDrive may lock the file — truncate it instead
            with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
                f.write("row_index,question,claim_text,papers_found,agree,"
                        "disagree,neutral,validity_score,agreement_rate,disagreement_rate\n")
            print(f"Reset: cleared {OUTPUT_CSV}")

    run_batch_evaluation()
