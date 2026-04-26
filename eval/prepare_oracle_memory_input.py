#!/usr/bin/env python3
"""
Prepares the retrieval input file for run_generation.py for the Oracle Memory Agent.

For every is_query=True turn in the dataset, the "retrieved evidence" is the
ground truth memory_used from the immediately following Assistant turn.
This simulates a perfect memory retrieval system.

Output is keyed by question text so that compute_llm_metrics_for_realmem.py
can match entries against its gt_map (also keyed by question text).

Usage:
    python eval/prepare_oracle_memory_input.py \
        --data_file dataset/Adeleke_Okonjo_dialogues_256k.json \
        --output_file eval/retrieval_result/Adeleke_Okonjo/oracle_memory_retrieval_input.json
"""

import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Build oracle memory retrieval input for run_generation.py."
    )
    parser.add_argument(
        "--data_file",
        required=True,
        help="Path to the dataset JSON (e.g. dataset/Adeleke_Okonjo_dialogues_256k.json).",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Where to write the retrieval input JSON.",
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_file}...")
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogues = data.get("dialogues", [])
    dialogues.sort(key=lambda d: (d.get("current_time", ""), d.get("session_identifier", "")))

    output = {}
    query_count = 0
    skipped_no_memory = 0

    for session in dialogues:
        turns = session.get("dialogue_turns", [])

        for turn_idx, turn in enumerate(turns):
            if not turn.get("is_query"):
                continue

            question = turn.get("content", "").strip()
            if not question:
                continue

            # Find the immediately following Assistant turn
            memory_used = []
            for j in range(turn_idx + 1, len(turns)):
                if turns[j].get("speaker") == "Assistant":
                    memory_used = turns[j].get("memory_used", [])
                    break

            if not memory_used:
                skipped_no_memory += 1

            # Pack each memory point as a separate chunk ranked_item
            ranked_items = [
                {
                    "res_type": "chunk",
                    "content": m.get("content", "").strip(),
                    "rank": i + 1,
                }
                for i, m in enumerate(memory_used)
                if m.get("content", "").strip()
            ]

            output[question] = {
                "question": question,
                "ranked_items": ranked_items,
            }
            query_count += 1

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote {query_count} query entries to {args.output_file}")
    if skipped_no_memory:
        print(f"  ({skipped_no_memory} queries had no memory_used in the following Assistant turn)")


if __name__ == "__main__":
    main()
