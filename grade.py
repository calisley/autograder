#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
from llm_grader import grade_async
from process_documents import process_all_documents, process_single_document
import asyncio

async def main():
    parser = argparse.ArgumentParser(
        description="Grade student submissions against a given answer key and optional rubric."
    )

    parser.add_argument(
        "submissions_folder",
        type=str,
        help="Path to the folder containing student PDF/DOCX submissions to be converted to Markdown."
    )
    parser.add_argument(
        "answer_key",
        type=str,
        help="Path to the answer key document (PDF/DOCX/etc.)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./grader_output.csv",
        help="Name of the CSV file to output grading results."
    )
    parser.add_argument(
        "--rubric",
        type=str,
        default=None,
        help="Optional path to the rubric document (PDF/DOCX/etc.)."
    )
    parser.add_argument(
        "--truncate",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of page numbers to exclude from PDF submissions."
    )

    args = parser.parse_args()

    # 1. Convert all submissions in the directory to Markdown
    print("Converting submissions to Markdown...")
    
    submissions = await process_all_documents(args.submissions_folder, "./submissions_markdown.csv","./submissions_backup", truncate=args.truncate)
    if submissions.empty:
        print(f"No valid documents found in {args.submissions_folder}")
        sys.exit(1)


    # 2. Convert the answer key document to Markdown
    if not os.path.isfile(args.answer_key):
        print(f"Answer key file {args.answer_key} does not exist.")
        sys.exit(1)
    print("Fetching answer key...")

    answer_key_md = await process_single_document(args.answer_key)
    if not answer_key_md:
        print(f"Failed to process answer key document: {args.answer_key}")
        sys.exit(1)

    # 3. Convert the rubric document to Markdown (if provided)
    if args.rubric:
        print("Fetching rubric...")

        if not os.path.isfile(args.rubric):
            print(f"Rubric file {args.rubric} does not exist. Proceeding without a rubric.")
            rubric_md = ""
        else:
            rubric_md = await process_single_document(args.rubric)
            if not rubric_md:
                print(f"Failed to process rubric document: {args.rubric}")
                rubric_md = ""
    else:
        # No rubric provided
        rubric_md = ""

    # 4. Call your async grading function to produce results
    print("Grading assignments...")
    results_df = await grade_async(
        df_submissions=submissions,
        answer_key_markdown=answer_key_md,
        rubric_markdown=rubric_md,
        model="o3-mini"
    )

    if results_df.empty:
        print("No grading results were returned. Check your grader logic.")
        sys.exit(1)

    # 5. Save results to CSV
    results_df.to_csv(args.output_csv, index=False)
    print(f"Grading complete! Results saved to {args.output_csv}")


if __name__ == "__main__":
    asyncio.run(main())
