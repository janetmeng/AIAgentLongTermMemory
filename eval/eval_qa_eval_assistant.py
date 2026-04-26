### NOT RIGHT!!!!!
# passed in all history dialogues into the judge eval LLM oops



# #!/usr/bin/env python3
"""
eval/eval_qa_eval_assistant.py

For every is_query=True turn in a RealMemBench dialogues file:
  1. Feed EvalAssistantAgent all previous session history (history_dialogue)
     plus any prior turns in the current session (conversation_history).
  2. Generate a candidate answer.
  3. Score with the QA judge prompt (0-3) from compute_llm_metrics_for_realmem.py.

The 4 QA judge inputs:
  1. User's current query
  2. User-related memory  → all previous dialogue history (what the agent can see)
  3. Reference answer     → oracle assistant's ground-truth response
  4. Candidate answer     → EvalAssistantAgent's generated response

Usage:
  python eval/eval_qa_eval_assistant.py \\
      --data_file dataset/Adeleke_Okonjo_dialogues_256k.json \\
      --output_file eval/results/Adeleke_Okonjo_eval_assistant_qa.json \\
      --base_url https://litellm.oit.duke.edu \\
      --assistant_model "gpt-5-mini" \\
      --judge_model "gpt-5-mini"
"""

import json
import os
import re
import sys
import argparse
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.multi_agent_dialogue_processor import EvalAssistantAgent, ConversationPhase
from utils.llm_client import LLMClient

# ── QA judge prompt (verbatim from compute_llm_metrics_for_realmem.py) ─────────
QA_EVAL_PROMPT = """Your task is to evaluate the consistency between the [candidate answer] and the [user-related memory].

You will be given four pieces of information:
1. The user's current query
2. The user-related memory, representing the latest valid user state
3. A reference answer based on the relevant memory
4. The candidate answer to be evaluated

Please follow these rules during evaluation:
- Focus only on whether "facts, constraints, preferences, and confirmed states" are correctly used
- Do NOT evaluate language style, tone, politeness, empathy, or fluency
- Do NOT give a high score just because the answer "sounds reasonable"
- The reference answer is only to help understand how relevant memory should ideally be used; a candidate answer does not need to exactly match the reference answer to receive a full score

Scoring criteria:
Score 0: Poor — the candidate answer conflicts with the user-related memory
Score 1: Fair — the candidate answer does not conflict with the relevant memory but is generic and not based on user memory
Score 2: Good — the candidate answer uses part of the user-related memory
Score 3: Very good — the candidate answer (like the reference answer) uses all of the user-related memory

Output format:
```json
{
    "score": int,
    "reason": str
}
```
"""


def parse_args():
    p = argparse.ArgumentParser(description="QA eval for EvalAssistantAgent on RealMemBench")
    p.add_argument("--data_file", default="dataset/Adeleke_Okonjo_dialogues_256k.json",
                   help="RealMemBench dialogues JSON file")
    p.add_argument("--output_file", default="eval/results/Adeleke_Okonjo_eval_assistant_qa.json",
                   help="Path to save results JSON")
    p.add_argument("--base_url", default="https://litellm.oit.duke.edu",
                   help="LiteLLM proxy base URL")
    p.add_argument("--assistant_model", default="gpt-5-mini",
                   help="Model for EvalAssistantAgent (the agent being evaluated)")
    p.add_argument("--judge_model", default="gpt-5-mini",
                   help="Model for the LLM judge")
    p.add_argument("--max_workers", type=int, default=4,
                   help="Parallel workers for LLM judge phase")
    return p.parse_args()


def extract_json(text: str):
    try:
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return None


def format_turns_as_text(turns: list) -> str:
    """Format a list of {speaker, content} dicts into readable text."""
    return "\n".join(f"{t['speaker']}: {t['content']}" for t in turns)


def run_judge(client, model: str, query: str, user_memory: str,
              reference_answer: str, candidate_answer: str):
    prompt = f"""{QA_EVAL_PROMPT}

### Input Data
1. Query: {query}
2. User-related Memory: {user_memory}
3. Reference Answer: {reference_answer}
4. Candidate Answer: {candidate_answer}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return extract_json(resp.choices[0].message.content)
    except Exception as e:
        print(f"  Judge error: {e}")
        return None


def main():
    args = parse_args()

    api_key = os.getenv("DUKE_LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"

    assistant_llm = LLMClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.assistant_model,
    )
    judge_client = OpenAI(api_key=api_key, base_url=args.base_url)

    print(f"Loading dataset: {args.data_file}")
    with open(args.data_file) as f:
        data = json.load(f)

    dialogues = data["dialogues"]
    print(f"Total sessions: {len(dialogues)}")

    # ── Phase 1: generate candidate answers ────────────────────────────────────
    # accumulated_history holds all turns from every completed session so far.
    # EvalAssistantAgent receives this as history_dialogue (cross-session context).
    accumulated_history: list = []

    items_to_judge = []

    for session in tqdm(dialogues, desc="Generating candidate answers"):
        turns = session["dialogue_turns"]
        current_time = session.get("current_time", "")
        session_id = session["session_identifier"]

        # Turns seen so far within this session (before the current query)
        session_turns_so_far: list = []

        i = 0
        while i < len(turns):
            turn = turns[i]

            if turn.get("is_query") and turn["speaker"] == "User":
                query_text = turn["content"]
                query_id = turn.get("query_id", f"{session_id}_t{i}")

                # Find the oracle assistant turn immediately after
                j = i + 1
                while j < len(turns) and turns[j]["speaker"] != "Assistant":
                    j += 1

                ref_answer = ""
                if j < len(turns):
                    ref_answer = turns[j].get("content", "")

                # Build user-related memory for the judge:
                # all history the agent can see = past sessions + current session so far
                history_text = format_turns_as_text(accumulated_history)
                session_text = format_turns_as_text(session_turns_so_far)
                if history_text and session_text:
                    user_memory_text = history_text + "\n\n" + session_text
                elif history_text:
                    user_memory_text = history_text
                elif session_text:
                    user_memory_text = session_text
                else:
                    user_memory_text = "No previous conversation history."

                # Instantiate EvalAssistantAgent with accumulated cross-session history
                agent = EvalAssistantAgent(
                    llm_client=assistant_llm,
                    history_dialogue=accumulated_history if accumulated_history else None,
                )

                # conversation_history = turns in THIS session before the query
                # (needs .speaker and .content attributes)
                conv_history = [
                    SimpleNamespace(speaker=t["speaker"], content=t["content"])
                    for t in session_turns_so_far
                ]

                # Generate candidate answer
                try:
                    candidate = agent.generate_response(
                        user_message=query_text,
                        conversation_history=conv_history,
                        memory_context="",                       # oracle — ignored
                        phase=ConversationPhase.EXPLORATION,     # oracle — ignored
                        current_time=current_time,
                        current_plan_items=None,                 # oracle — ignored
                        current_goal="",                         # oracle — ignored
                    )
                except Exception as e:
                    print(f"  Agent error [{query_id}]: {e}")
                    candidate = ""

                items_to_judge.append({
                    "query_id": query_id,
                    "session_identifier": session_id,
                    "query": query_text,
                    "reference_answer": ref_answer,
                    "candidate_answer": candidate,
                    "user_memory_text": user_memory_text,
                    "current_time": current_time,
                    "topic": turn.get("topic", ""),
                    "category_name": turn.get("category_name", ""),
                })

                # Add the query + candidate to current session history so subsequent
                # queries in the same session see the agent's own prior answers.
                session_turns_so_far.append({"speaker": "User", "content": query_text})
                session_turns_so_far.append({"speaker": "Assistant", "content": candidate})

                # Skip past the oracle assistant turn
                i = j + 1
                continue

            # Non-query turn: accumulate for current session context
            session_turns_so_far.append({"speaker": turn["speaker"], "content": turn["content"]})
            i += 1

        # After session ends, add the oracle turns to accumulated history
        # so future sessions have the ground-truth history as context.
        accumulated_history.extend(
            {"speaker": t["speaker"], "content": t["content"]}
            for t in turns
        )

    print(f"\nGenerated {len(items_to_judge)} candidate answers.")

    # ── Phase 2: LLM judge (parallel) ──────────────────────────────────────────
    print(f"Running LLM judge with {args.max_workers} workers...")

    def judge_item(item):
        result = run_judge(
            judge_client,
            args.judge_model,
            item["query"],
            item["user_memory_text"],
            item["reference_answer"],
            item["candidate_answer"],
        )
        return item["query_id"], item, result

    results = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(judge_item, item): item["query_id"] for item in items_to_judge}
        for future in tqdm(as_completed(futures), total=len(items_to_judge), desc="Judging"):
            qid, item, judge_result = future.result()
            results[qid] = {**item, "judge_result": judge_result}

    # ── Phase 3: aggregate ──────────────────────────────────────────────────────
    qa_scores = [
        r["judge_result"]["score"]
        for r in results.values()
        if r.get("judge_result") and isinstance(r["judge_result"].get("score"), (int, float))
    ]

    summary = {}
    if qa_scores:
        avg = float(np.mean(qa_scores))
        dist = {i: qa_scores.count(i) for i in range(4)}
        summary = {
            "average_qa_score": round(avg, 4),
            "qa_score_distribution": dist,
            "qa_hallucination_rate": round(dist.get(0, 0) / len(qa_scores), 4),
            "qa_perfect_rate": round(dist.get(3, 0) / len(qa_scores), 4),
            "total_evaluated": len(qa_scores),
            "assistant_model": args.assistant_model,
            "judge_model": args.judge_model,
        }
        print(f"\n{'='*50}")
        print(f"Average QA Score : {avg:.4f} / 3")
        print(f"Distribution     : {dist}")
        print(f"Hallucination    : {summary['qa_hallucination_rate']:.4f}  (score=0)")
        print(f"Perfect          : {summary['qa_perfect_rate']:.4f}  (score=3)")
        print(f"Total evaluated  : {len(qa_scores)}")
    else:
        print("No scores produced — check errors above.")

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output = {"summary": summary, "detailed_results": results}
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
