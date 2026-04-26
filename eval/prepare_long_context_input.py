#!/usr/bin/env python3
"""
Prepares the retrieval input file for run_generation.py for the Long Context Agent.

For every is_query=True turn in the dataset, the "retrieved evidence" is the
full dialogue history up to that point (all prior sessions + current-session
turns before the query). This is packed as a single chunk in ranked_items so
that run_generation.py can consume it unchanged.

Output is keyed by question text so that compute_llm_metrics_for_realmem.py
can match entries against its gt_map (which is also keyed by question text).

Usage:
    python eval/prepare_long_context_input.py \
        --data_file dataset/Adeleke_Okonjo_dialogues_256k.json \
        --output_file eval/retrieval_result/Adeleke_Okonjo/long_context_retrieval_input.json
"""

import json
import os
import argparse
from pathlib import Path


def format_history(sessions_before, turns_before_in_current):
    """Combine all prior session turns and current-session pre-query turns."""
    lines = []
    for session in sessions_before:
        session_id = session.get("session_identifier", "unknown")
        time_str = session.get("current_time", "")
        header = f"[Session {session_id}" + (f" - {time_str}" if time_str else "") + "]"
        lines.append(header)
        for turn in session.get("dialogue_turns", []):
            lines.append(f"{turn.get('speaker', 'Unknown')}: {turn.get('content', '')}")
        lines.append("")

    if turns_before_in_current:
        lines.append("[Current Session]")
        for turn in turns_before_in_current:
            lines.append(f"{turn.get('speaker', 'Unknown')}: {turn.get('content', '')}")

    return "\n".join(lines).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Build long-context retrieval input for run_generation.py."
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
    # Sort chronologically by current_time, falling back to session_identifier
    dialogues.sort(key=lambda d: (d.get("current_time", ""), d.get("session_identifier", "")))

    output = {}
    query_count = 0

    for session_idx, session in enumerate(dialogues):
        turns = session.get("dialogue_turns", [])
        sessions_before = dialogues[:session_idx]

        for turn_idx, turn in enumerate(turns):
            if not turn.get("is_query"):
                continue

            question = turn.get("content", "").strip()
            if not question:
                continue

            history_text = format_history(sessions_before, turns[:turn_idx])

            # Pack history as a single chunk — run_generation.py filters by res_type="chunk"
            output[question] = {
                "question": question,
                "ranked_items": [
                    {
                        "res_type": "chunk",
                        "content": history_text if history_text else "(No prior history)",
                        "rank": 1,
                    }
                ],
            }
            query_count += 1

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote {query_count} query entries to {args.output_file}")


if __name__ == "__main__":
    main()
