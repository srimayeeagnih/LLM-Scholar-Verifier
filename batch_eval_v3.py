"""
batch_eval.py
─────────────
Batch evaluation script for MCP_test_v3.

Reads answers from the Excel dataset, runs each through the full pipeline,
and writes validity scores to a checkpoint CSV after every row.

Stop anytime with Ctrl+C — re-run to resume from where you left off.

Usage:
    python batch_eval.py                              # Statistics and Data Analytics sheet
    python batch_eval.py "Machine Learning"           # Machine Learning sheet
    python batch_eval.py "Statistics and Data Analytics"
"""

import sys
import os
import time
import traceback
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
EXCEL_PATH            = r"C:\Users\srima\OneDrive\Desktop\INSY669\Group Project\Question Bank with LLM answers.xlsx"
ANSWER_COL            = "Answers"
QUESTION_COL          = "Questions"

# Sheet name can be passed as a command-line argument; defaults to Statistics sheet
SHEET_NAME = sys.argv[1] if len(sys.argv) > 1 else "Statistics and Data Analytics"

# All results go into one Excel file; each sheet run gets its own sheet tab
_safe_sheet  = SHEET_NAME.replace(" ", "_").replace("&", "and")
OUTPUT_XLSX  = r"C:\Users\srima\OneDrive\Desktop\INSY669\Group Project\batch_results_Statistics_and_Data_Analytics.csv.xlsx"
OUTPUT_SHEET = "Version 5"

BATCH_SIZE            = 10   # rows per batch
SLEEP_BETWEEN_ROWS    = 5    # seconds between rows (rate limiting)
SLEEP_BETWEEN_BATCHES = 60   # seconds between batches (rate limiting)
MAX_RESULTS_PER_SOURCE = 10  # keep small for speed; increase for more coverage
# ─────────────────────────────────────────────────────────────────────────────

# Import pipeline functions from MCP_test_v2 (must be in the same folder)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
from MCP_test_v2 import search_by_claims, compute_claim_verdict


COLUMNS = [
    "row_index", "question", "claim_text",
    "papers_found", "agree", "disagree", "neutral",
    "validity_score", "agreement_rate", "disagreement_rate",
]


def load_done_claims(output_xlsx, output_sheet):
    """Return the set of claim texts already written to the checkpoint sheet."""
    if not os.path.exists(output_xlsx):
        return set()
    try:
        xl = pd.ExcelFile(output_xlsx)
        if output_sheet not in xl.sheet_names:
            return set()
        existing = xl.parse(output_sheet)
        return set(existing["claim_text"].dropna().tolist())
    except Exception:
        return set()


def init_output_xlsx(output_xlsx, output_sheet):
    """Create the Excel file / sheet with headers if it doesn't exist yet."""
    empty = pd.DataFrame(columns=COLUMNS)
    if not os.path.exists(output_xlsx):
        empty.to_excel(output_xlsx, sheet_name=output_sheet, index=False)
    else:
        xl = pd.ExcelFile(output_xlsx)
        if output_sheet not in xl.sheet_names:
            with pd.ExcelWriter(output_xlsx, engine="openpyxl", mode="a") as writer:
                empty.to_excel(writer, sheet_name=output_sheet, index=False)


def append_result(output_xlsx, output_sheet, row_dict):
    """Append a single result row to the checkpoint sheet."""
    existing = pd.read_excel(output_xlsx, sheet_name=output_sheet)
    updated  = pd.concat([existing, pd.DataFrame([row_dict])], ignore_index=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        updated.to_excel(writer, sheet_name=output_sheet, index=False)


def run_batch_evaluation():
    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset from:\n  {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    print(f"Loaded {len(df)} rows from sheet '{SHEET_NAME}'")

    # ── Resume support ────────────────────────────────────────────────────────
    done_claims = load_done_claims(OUTPUT_XLSX, OUTPUT_SHEET)
    print(f"Already processed: {len(done_claims)} row(s) — will skip these.")
    remaining = len(df) - len(done_claims)
    print(f"Remaining:         {remaining} row(s)\n")

    init_output_xlsx(OUTPUT_XLSX, OUTPUT_SHEET)

    completed = 0
    skipped   = 0

    try:
        for batch_start in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[batch_start : batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"\n{'#'*70}")
            print(f"# Batch {batch_num} / {total_batches}  "
                  f"(rows {batch_start + 1}–{min(batch_start + BATCH_SIZE, len(df))})")
            print(f"{'#'*70}")

            rows_in_batch = list(batch.iterrows())
            for local_i, (abs_i, row) in enumerate(rows_in_batch):
                claim_text = str(row.get(ANSWER_COL, "")).strip()
                question   = str(row.get(QUESTION_COL, "")).strip()

                # Skip empty rows
                if not claim_text or claim_text.lower() == "nan":
                    print(f"\n[Row {abs_i + 1}/{len(df)}] Empty answer — skipping.")
                    skipped += 1
                    continue

                # Skip already-processed rows (resume support)
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
                    claim_results = search_by_claims(claim_text, MAX_RESULTS_PER_SOURCE)

                    # Aggregate verdicts across all sub-claims in the answer
                    total_agree    = 0
                    total_disagree = 0
                    total_neutral  = 0
                    total_papers   = 0

                    for claim_data in claim_results:
                        verdict = compute_claim_verdict(claim_data)
                        ct = verdict["claim_totals"]
                        total_agree    += ct.get("agree", 0)
                        total_disagree += ct.get("disagree", 0)
                        total_neutral  += ct.get("neutral", ct.get("missing", 0))
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

                    append_result(OUTPUT_XLSX, OUTPUT_SHEET, result)
                    done_claims.add(claim_text)
                    completed += 1

                    print(f"\n  Validity score: {validity_score:.1%}  "
                          f"(agree={total_agree}  disagree={total_disagree}  "
                          f"neutral={total_neutral}  papers={total_papers})")
                    print(f"  Saved to {OUTPUT_XLSX} (sheet: {OUTPUT_SHEET})")

                except Exception as e:
                    print(f"\n  ERROR on row {abs_i + 1}: {e}")
                    traceback.print_exc(file=sys.stdout)
                    print("  Row will be retried on next run.")
                    # Do NOT add to done_claims so it retries on resume

                # Sleep between rows (except after the last row in the batch)
                if local_i < len(rows_in_batch) - 1:
                    print(f"\n  Waiting {SLEEP_BETWEEN_ROWS}s...")
                    time.sleep(SLEEP_BETWEEN_ROWS)

            # Sleep between batches (except after the final batch)
            if batch_start + BATCH_SIZE < len(df):
                print(f"\n--- Batch {batch_num} complete. "
                      f"Waiting {SLEEP_BETWEEN_BATCHES}s before next batch ---")
                time.sleep(SLEEP_BETWEEN_BATCHES)

    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print("Stopped by user (Ctrl+C).")
        print(f"  Completed this session : {completed} row(s)")
        print(f"  Skipped (already done) : {skipped} row(s)")
        print(f"  Results saved to       : {OUTPUT_XLSX} (sheet: {OUTPUT_SHEET})")
        print("  Re-run this script to resume from where you left off.")
        print(f"{'='*70}")
        sys.exit(0)

    # ── All done ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("All rows processed!")
    print(f"  Completed : {completed} row(s)")
    print(f"  Skipped   : {skipped} row(s)")
    print(f"  Output    : {OUTPUT_XLSX} (sheet: {OUTPUT_SHEET})")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_batch_evaluation()
