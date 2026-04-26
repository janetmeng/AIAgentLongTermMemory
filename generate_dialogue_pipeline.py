#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dialogue Generation Pipeline

Generates a complete series of dialogues for a given persona and topic by
orchestrating the five-stage pipeline:
  1. ProjectOutlineProcessor       -> project blueprint
  2. EventProcessor                -> event sequences with volatility
  3. SummaryProcessor              -> per-event session summaries
  4. MultiAgentDialogueProcessor   -> final dialogues with memory extraction
  5. DialoguePostprocessor         -> merge & annotate (is_query, query_id,
                                     session_uuid, extracted_memory, cleaned XML)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pypinyin import Style, lazy_pinyin

from pipeline.event_processor import EventProcessor
from pipeline.multi_agent_dialogue_processor import (
    ConversationController,
    MultiAgentDialogueProcessor,
)
from pipeline.project_outline_processor import ProjectOutlineProcessor
from pipeline.summary_processor import SummaryProcessor
from utils.dialogue_postprocessor import DialoguePostprocessor
from utils.llm_client import LLMClient, create_client

load_dotenv()

API_KEY = os.getenv("DUKE_LITELLM_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://litellm.oit.duke.edu")
DEFAULT_MODEL = "gpt-5-mini"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def convert_to_safe_filename(text: str) -> str:
    """Convert text to a filesystem-safe name using pinyin transliteration."""
    if not text:
        return "unknown"
    pinyin_list = lazy_pinyin(text, style=Style.NORMAL, neutral_tone_with_five=True)
    pinyin_text = "".join(pinyin_list)
    safe_name = "".join(c if c.isalnum() else "_" for c in pinyin_text)
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    return safe_name.strip("_") or "unknown"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path}")


def load_data_files(
    persona_file: str = "dataset/all_persona_topic/persona_all.json",
    topic_attr_file: str = "dataset/all_persona_topic/topic&goal&attr.json",
    person_goal_file: str = "dataset/all_persona_topic/person&goal.json",
):
    personas = load_json(Path(persona_file))
    topics = load_json(Path(topic_attr_file))
    person_goals = load_json(Path(person_goal_file))
    return personas, topics, person_goals


def find_persona(personas: List[Dict], name: str) -> Optional[Dict]:
    for p in personas:
        if p.get("name") == name:
            return p
    return None


def find_topic(topics_data: Dict, topic_id: str) -> Optional[Dict]:
    for t in topics_data.get("topics", []):
        if t.get("topic_id") == topic_id:
            return t
    return None


def find_task(person_goals: List[Dict], person_name: str, topic_id: str, task_id=None):
    """Find a specific assigned task for a person.

    If *task_id* is provided the match is exact. Otherwise the first task
    matching *topic_id* is returned.
    """
    for entry in person_goals:
        if entry.get("persona_name") != person_name:
            continue
        for task in entry.get("assigned_tasks", []):
            if task.get("topic_id") != topic_id:
                continue
            if task_id is not None and str(task.get("task_id")) != str(task_id):
                continue
            return task
    return None


def date_to_weekday(date_str: str) -> str:
    if not date_str:
        return ""
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return ["Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"][
                datetime.strptime(date_str, fmt).weekday()
            ]
        except ValueError:
            continue
    return date_str


# ---------------------------------------------------------------------------
# Stage 1 – Project Blueprint
# ---------------------------------------------------------------------------

def stage_blueprint(
    person_data: Dict,
    project_attributes: List[str],
    project_goal: str,
    topic_name: str,
    task_id,
    model: str = DEFAULT_MODEL,
) -> Optional[Dict]:
    """Generate a project blueprint via ProjectOutlineProcessor."""
    print("\n=== Stage 1: Generating Project Blueprint ===")

    llm_client = create_client(api_key=API_KEY, base_url=BASE_URL, model=model)
    processor = ProjectOutlineProcessor(
        llm_client=llm_client,
        checkpoint_dir="output/checkpoints/project_outline",
    )

    processor_input = {
        "persona": person_data,
        "project_attributes": project_attributes,
        "primary_goal": project_goal,
    }

    result = processor.process(data=processor_input, use_checkpoint=False)
    if not result.success:
        print(f"  FAILED: {result.error_message}")
        return None

    blueprint = result.data
    # Attach metadata used by downstream stages
    blueprint["_metadata"] = {
        "person_name": person_data.get("name", "unknown"),
        "persona_role": person_data.get("role", "Unknown"),
        "total_attributes_used": len(project_attributes),
    }
    blueprint["selected_topic"] = topic_name
    blueprint["selected_goal"] = project_goal
    blueprint["selected_task_id"] = task_id
    if project_attributes:
        blueprint["project_attributes_schema"] = project_attributes

    # Persist
    safe_person = convert_to_safe_filename(person_data.get("name", "unknown"))
    safe_topic = convert_to_safe_filename(topic_name)
    project_id = f"{safe_topic}_{task_id}"
    out_path = Path(f"output/{safe_person}/{project_id}/project_blueprints/{project_id}_blueprint.json")
    save_json(blueprint, out_path)

    print(f"  Blueprint generated with {len(project_attributes)} attributes.")
    return blueprint


# ---------------------------------------------------------------------------
# Stage 2 – Event Sequences
# ---------------------------------------------------------------------------

def stage_events(
    person_data: Dict,
    blueprint: Dict,
    project_attributes: List[str],
    model: str = DEFAULT_MODEL,
) -> Optional[Dict]:
    """Generate event sequences via EventProcessor."""
    print("\n=== Stage 2: Generating Event Sequences ===")

    llm_client = create_client(api_key=API_KEY, base_url=BASE_URL, model=model)
    event_processor = EventProcessor(
        llm_client=llm_client,
        checkpoint_dir="output/checkpoints/events",
    )

    processor_input = {
        "user_profile": person_data,
        "project_state": {
            "current_stage": "planning",
            "progress": 0,
            "milestones_completed": [],
            "current_focus": blueprint.get("project_goal", ""),
        },
        "project_blueprint": blueprint,
    }

    result = event_processor.process(data=processor_input, use_checkpoint=False)
    if not result.success:
        print(f"  FAILED: {result.error_message}")
        return None

    raw = result.data
    person_name = person_data.get("name", "unknown")

    # Normalise into {"events": [...], "_metadata": {...}}
    if isinstance(raw, list):
        events_data = {"events": raw}
    elif isinstance(raw, dict):
        events_data = raw if "events" in raw else {"events": [raw]}
    else:
        print(f"  FAILED: Unexpected events data format: {type(raw)}")
        return None

    events_data["_metadata"] = {
        "person_name": person_name,
        "persona_role": person_data.get("role", "Unknown"),
        "blueprint_topic": blueprint.get("selected_topic", ""),
        "total_attributes_used": len(project_attributes),
        "event_count": len(events_data.get("events", [])),
    }

    # Persist
    safe_person = convert_to_safe_filename(person_name)
    safe_topic = convert_to_safe_filename(blueprint.get("selected_topic", "unknown"))
    task_id = blueprint.get("selected_task_id", "unknown")
    project_id = f"{safe_topic}_{task_id}"
    out_path = Path(f"output/{safe_person}/{project_id}/project_events/{project_id}_events.json")
    save_json(events_data, out_path)

    num_events = len(events_data.get("events", []))
    print(f"  Generated {num_events} events.")
    return events_data


# ---------------------------------------------------------------------------
# Stage 3 – Session Summaries (per event)
# ---------------------------------------------------------------------------

def stage_summaries(
    person_data: Dict,
    blueprint: Dict,
    events_data: Dict,
    model: str = DEFAULT_MODEL,
) -> Optional[Dict]:
    """Generate session summaries for each event via SummaryProcessor."""
    print("\n=== Stage 3: Generating Session Summaries ===")

    events_list = events_data.get("events", [])
    if not events_list:
        print("  FAILED: No events to summarise.")
        return None

    llm_client = create_client(api_key=API_KEY, base_url=BASE_URL, model=model)
    summary_processor = SummaryProcessor(
        llm_client=llm_client,
        checkpoint_dir="output/checkpoints/summaries",
    )

    all_summaries: List[Dict] = []

    for idx, target_event in enumerate(events_list):
        print(f"  Summarising event {idx + 1}/{len(events_list)}: "
              f"{target_event.get('event_name', 'Unknown')}")

        input_data = {
            "user_profile": person_data,
            "project_blueprint": blueprint,
            "full_event_log": events_list,
            "target_event": target_event,
        }

        result = summary_processor.process(input_data, use_checkpoint=False)
        if not result.success:
            print(f"    FAILED: {result.error_message}")
            continue

        output = result.data
        if isinstance(output, list):
            sessions = output
        elif isinstance(output, dict):
            if "sessions" in output and isinstance(output["sessions"], list):
                sessions = output["sessions"]
            elif "session_id" in output:
                sessions = [output]
            else:
                sessions = []
        else:
            sessions = []

        all_summaries.extend(sessions)
        print(f"    -> {len(sessions)} session summaries produced.")

    if not all_summaries:
        print("  FAILED: No session summaries generated.")
        return None

    person_name = person_data.get("name", "unknown")
    summary_output = {
        "_metadata": {
            "person_name": person_name,
            "persona_role": person_data.get("role", "Unknown"),
            "blueprint_topic": blueprint.get("selected_topic", ""),
            "total_events": len(events_list),
            "total_sessions": len(all_summaries),
        },
        "sessions": all_summaries,
    }

    # Persist
    safe_person = convert_to_safe_filename(person_name)
    safe_topic = convert_to_safe_filename(blueprint.get("selected_topic", "unknown"))
    task_id = blueprint.get("selected_task_id", "unknown")
    project_id = f"{safe_topic}_{task_id}"
    out_path = Path(f"output/{safe_person}/{project_id}/session_summaries/{project_id}_summary.json")
    save_json(summary_output, out_path)

    print(f"  Total session summaries: {len(all_summaries)}")
    return summary_output


# ---------------------------------------------------------------------------
# Stage 4 – Multi-Agent Dialogue Generation
# ---------------------------------------------------------------------------

def _get_event_session_summaries(
    event_id, all_sessions: List[Dict]
) -> List[Dict]:
    """Return session summaries belonging to the same event."""
    return [s for s in all_sessions if s.get("event_id") == event_id]


def _get_history_dialogue(
    processed_sessions: List[str], output_dir: Path
) -> List[Dict]:
    """Load the last two turns of each prior session as context."""
    history: List[Dict] = []
    for sid in processed_sessions:
        fpath = output_dir / f"{sid}.json"
        if not fpath.exists():
            continue
        try:
            data = load_json(fpath)
            turns = data.get("dialogue_turns", [])
            if turns:
                history.extend(turns[-2:])
        except Exception:
            continue
    return history


def stage_dialogues(
    person_data: Dict,
    blueprint: Dict,
    events_data: Dict,
    summary_data: Dict,
    dialogue_model: str = DEFAULT_MODEL,
    evaluation_model: str = DEFAULT_MODEL,
    memory_model: str = DEFAULT_MODEL,
    memory_retrieve_model: str = DEFAULT_MODEL,
    dedup_model: str = DEFAULT_MODEL,
    semantic_schedule_model: str = DEFAULT_MODEL,
    max_turns: int = 24,
    max_retries: int = 2,
) -> Optional[Dict]:
    """Generate dialogues for every session via MultiAgentDialogueProcessor.

    For each session the pipeline:
      - Retrieves relevant memories
      - Runs multi-turn conversation with UserAgent + AssistantAgent
      - Evaluates goal completion
      - Extracts new memory points and deduplicates
      - Persists dialogue + updated memory
    """
    print("\n=== Stage 4: Generating Multi-Agent Dialogues ===")

    sessions = summary_data.get("sessions", [])
    if not sessions:
        print("  FAILED: No session summaries to generate dialogues for.")
        return None

    person_name = person_data.get("name", "unknown")
    safe_person = convert_to_safe_filename(person_name)
    safe_topic = convert_to_safe_filename(blueprint.get("selected_topic", "unknown"))
    task_id = blueprint.get("selected_task_id", "unknown")
    project_id = f"{safe_topic}_{task_id}"

    output_dir = Path(f"output/{safe_person}/{project_id}/dialogues")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build event list
    event_list = events_data.get("events", [])

    # Create LLM clients for each agent role
    dialogue_client = create_client(api_key=API_KEY, base_url=BASE_URL, model=dialogue_model)
    evaluation_client = create_client(api_key=API_KEY, base_url=BASE_URL, model=evaluation_model)
    memory_client = create_client(api_key=API_KEY, base_url=BASE_URL, model=memory_model)
    memory_retrieve_client = create_client(api_key=API_KEY, base_url=BASE_URL, model=memory_retrieve_model)
    dedup_client = create_client(api_key=API_KEY, base_url=BASE_URL, model=dedup_model)
    semantic_schedule_client = create_client(api_key=API_KEY, base_url=BASE_URL, model=semantic_schedule_model)

    # Instantiate the processor (inherits BaseProcessor)
    processor = MultiAgentDialogueProcessor(dialogue_client)

    # Wire up a ConversationController with dedicated clients
    controller = ConversationController(
        dialogue_client=dialogue_client,
        evaluation_client=evaluation_client,
        memory_client=memory_client,
        memory_retrieve_client=memory_retrieve_client,
        dedup_client=dedup_client,
        semantic_schedule_client=semantic_schedule_client,
        project_attributes_schema=blueprint.get("project_attributes_schema", ""),
    )
    processor.conversation_controller = controller

    # Person-level memory accumulates across sessions
    person_memory: Dict[str, Any] = {
        "memory_points": [],
        "total_sessions": 0,
        "last_updated": time.time(),
        "metadata": {"person_name": person_name},
    }

    # Try loading existing memory
    memory_file = Path(f"output/{safe_person}/{safe_person}_memory.json")
    if memory_file.exists():
        try:
            person_memory = load_json(memory_file)
            print(f"  Loaded existing memory: {len(person_memory.get('memory_points', []))} points")
        except Exception:
            pass

    # Schedule / plan state
    schedule_file = Path(f"output/{safe_person}/schedule.json")
    plan_items: List[Dict] = []
    current_date = datetime.now().strftime("%Y-%m-%d")
    if schedule_file.exists():
        try:
            sched = load_json(schedule_file)
            plan_items = sched.get("plan_items", [])
            current_date = sched.get("current_date", current_date) or current_date
        except Exception:
            pass

    processed_sessions: List[str] = []
    all_dialogues: List[Dict] = []

    for session in sessions:
        session_id = session.get("session_id", f"S{len(processed_sessions)+1:03d}")
        print(f"\n  --- Session {session_id} ---")

        # Gather context
        current_event_summaries = _get_event_session_summaries(
            session.get("event_id", 0), sessions
        )
        history_dialogue = _get_history_dialogue(processed_sessions, output_dir)
        current_time_str = f"{current_date} ({date_to_weekday(current_date)})"

        input_data = {
            "user_input_profile": person_data,
            "full_event_log": event_list,
            "current_event_session_summary_list": current_event_summaries,
            "target_session": {
                "session_id": session_id,
                "session_summary": session.get("session_summary", ""),
            },
            "max_turns": max_turns,
            "memory_context": person_memory,
            "history_dialogue": history_dialogue,
            "current_time": current_time_str,
            "current_plan_items": plan_items,
        }

        # Retry loop
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"    Retry {attempt}/{max_retries} ...")
                    time.sleep(attempt)

                result = processor.process(input_data, use_checkpoint=False)

                if not result.success:
                    last_error = result.error_message
                    print(f"    Attempt failed: {last_error}")
                    continue

                dialogue_output = result.data

                # --- Memory extraction + dedup ---
                retrieved_memory_data = dialogue_output.get("retrieved_memory", {})
                retrieved_memory_points = retrieved_memory_data.get("all", [])
                new_memory_points: List[Dict] = []

                dialogue_data_for_memory = {
                    "session_id": session_id,
                    "dialogue_turns": dialogue_output.get("dialogue_turns", []),
                }

                try:
                    extracted_memory = controller.extract_session_memory(dialogue_data_for_memory)
                    new_memory_points = extracted_memory.get("memory_points", [])

                    # Tag new memory with project id
                    for pt in new_memory_points:
                        idx = pt.get("index", "")
                        if idx and not idx.startswith(f"{project_id}-"):
                            pt["index"] = f"{project_id}-{idx}"

                    # Deduplicate combined pool
                    combined = retrieved_memory_points + new_memory_points
                    if combined:
                        dedup_result = controller.deduplicate_memory_points({"memory_points": combined})
                        final_points = dedup_result.get("memory_points", [])

                        # Merge back into person memory
                        global_index_map = {
                            p.get("index"): i
                            for i, p in enumerate(person_memory["memory_points"])
                        }
                        for dp in final_points:
                            pidx = dp.get("index")
                            if not pidx:
                                continue
                            if pidx in global_index_map:
                                gi = global_index_map[pidx]
                                person_memory["memory_points"][gi]["content"] = dp.get("content", "")
                                person_memory["memory_points"][gi]["discard"] = dp.get("discard", False)
                            else:
                                person_memory["memory_points"].append(dp)
                                global_index_map[pidx] = len(person_memory["memory_points"]) - 1

                        active_count = len([p for p in final_points if not p.get("discard", False)])
                        print(f"    Memory: {len(combined)} -> {active_count} active points")
                except Exception as mem_err:
                    print(f"    Memory extraction error: {mem_err}")

                # --- Update plan items ---
                updated_plan = dialogue_output.get("updated_plan_items", [])
                if updated_plan:
                    plan_items = updated_plan

                # --- Save dialogue ---
                dialogue_record = {
                    "session_id": session_id,
                    "input_data": input_data,
                    "dialogue_turns": dialogue_output.get("dialogue_turns", []),
                    "current_time": current_time_str,
                    "goal_evaluation": dialogue_output.get("goal_evaluation", {}),
                    "new_memory_points": new_memory_points,
                    "retrieved_memory_points": retrieved_memory_points,
                    "metadata": {
                        "total_turns": len(dialogue_output.get("dialogue_turns", [])),
                        "person_name": person_name,
                        "processing_time": dialogue_output.get("metadata", {}).get("processing_time", 0),
                    },
                }
                save_json(dialogue_record, output_dir / f"{session_id}.json")

                # Persist person memory after each session
                person_memory["total_sessions"] = len(processed_sessions) + 1
                person_memory["last_updated"] = time.time()
                save_json(person_memory, memory_file)

                # Save project-level memory snapshot
                proj_mem_path = Path(
                    f"output/{safe_person}/{project_id}/project_memories/{project_id}_memory.json"
                )
                save_json(person_memory, proj_mem_path)

                all_dialogues.append(dialogue_record)
                processed_sessions.append(session_id)

                turns = len(dialogue_output.get("dialogue_turns", []))
                score = dialogue_output.get("goal_evaluation", {}).get("overall_score", 0)
                print(f"    Completed: {turns} turns, goal score: {score:.0f}")

                last_error = None
                break  # success

            except Exception as exc:
                last_error = str(exc)
                print(f"    Exception on attempt {attempt}: {last_error}")

        if last_error is not None:
            print(f"    SKIPPED session {session_id} after {max_retries + 1} attempts.")

        # Advance simulated date by 1-3 days between sessions
        import random
        current_date = (
            datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=random.randint(1, 3))
        ).strftime("%Y-%m-%d")

    # Save schedule
    sched_data = {
        "plan_items": plan_items,
        "current_date": current_date,
        "metadata": {"person_name": person_name, "updated_time": time.time()},
    }
    save_json(sched_data, schedule_file)

    if all_dialogues:
        print(f"\n  Successfully generated {len(all_dialogues)}/{len(sessions)} dialogues.")
        return {
            "sessions": all_dialogues,
            "total_generated": len(all_dialogues),
            "processed_sessions": processed_sessions,
        }

    print("\n  FAILED: No dialogues were generated.")
    return None


# ---------------------------------------------------------------------------
# Stage 5 – Post-processing (merge + is_query + query_id + session_uuid)
# ---------------------------------------------------------------------------

def stage_postprocess(
    person_data: Dict,
    blueprint: Dict,
    processed_sessions: List[str],
    output_dir: str = "output",
) -> Optional[str]:
    """Post-process raw dialogues into the final dataset format.

    Creates the ``interleaved_dialogue_queue_state.json`` expected by
    ``DialoguePostprocessor.merge_person_dialogues()`` and then runs the
    merge.  The result is a single JSON file with:
      - ``is_query`` / ``query_id`` on each dialogue turn
      - ``session_uuid`` per session
      - ``extracted_memory`` per session
      - cleaned content (XML tags stripped)
    """
    print("\n=== Stage 5: Post-processing & Merging Dialogues ===")

    if not processed_sessions:
        print("  FAILED: No processed sessions to post-process.")
        return None

    person_name = person_data.get("name", "unknown")
    safe_person = convert_to_safe_filename(person_name)
    safe_topic = convert_to_safe_filename(blueprint.get("selected_topic", "unknown"))
    task_id = blueprint.get("selected_task_id", "unknown")
    project_id = f"{safe_topic}_{task_id}"

    person_dir = Path(output_dir) / safe_person

    # Build session identifiers in the format "project_id:session_id"
    session_identifiers = [f"{project_id}:{sid}" for sid in processed_sessions]

    # Create the queue-state file that merge_person_dialogues() expects
    queue_state = {
        "status": "completed",
        "processed_session_ids": session_identifiers,
    }
    queue_state_path = person_dir / "interleaved_dialogue_queue_state.json"
    save_json(queue_state, queue_state_path)

    # Run the postprocessor merge.
    # Override the postprocessor's convert_to_safe_filename so it matches the
    # pypinyin-based version used by this pipeline to create output directories.
    postprocessor = DialoguePostprocessor()
    postprocessor.convert_to_safe_filename = convert_to_safe_filename
    try:
        output_filename = f"{safe_person}_dialogues_{project_id}.json"
        merged_path = postprocessor.merge_person_dialogues(
            person_name=person_name,
            output_dir=output_dir,
            output_filename=output_filename,
        )
        print(f"  Merged dialogues saved to: {merged_path}")
        return merged_path
    except Exception as exc:
        print(f"  FAILED: Post-processing error: {exc}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    person_name: str,
    topic_id: str,
    task_id=None,
    blueprint_model: str = DEFAULT_MODEL,
    event_model: str = DEFAULT_MODEL,
    summary_model: str = DEFAULT_MODEL,
    dialogue_model: str = DEFAULT_MODEL,
    evaluation_model: str = DEFAULT_MODEL,
    memory_model: str = DEFAULT_MODEL,
    memory_retrieve_model: str = DEFAULT_MODEL,
    dedup_model: str = DEFAULT_MODEL,
    semantic_schedule_model: str = DEFAULT_MODEL,
    max_turns: int = 24,
    max_retries: int = 2,
):
    """Run the full five-stage dialogue generation pipeline."""
    personas, topics_data, person_goals = load_data_files()

    # Resolve persona
    person_data = find_persona(personas, person_name)
    if person_data is None:
        print(f"Persona '{person_name}' not found. Available:")
        for p in personas:
            print(f"  - {p.get('name')}")
        sys.exit(1)

    # Resolve topic
    topic = find_topic(topics_data, topic_id)
    if topic is None:
        print(f"Topic '{topic_id}' not found. Available:")
        for t in topics_data.get("topics", []):
            print(f"  - {t.get('topic_id')}: {t.get('topic_name')}")
        sys.exit(1)

    topic_name = topic["topic_name"]
    project_attributes = topic.get("project_attributes", [])

    # Resolve task / goal
    task = find_task(person_goals, person_name, topic_id, task_id)
    if task is None:
        print(f"No assigned task found for {person_name} with topic '{topic_id}'"
              + (f" task_id={task_id}" if task_id else ""))
        print("Available tasks for this persona:")
        for entry in person_goals:
            if entry.get("persona_name") == person_name:
                for t in entry.get("assigned_tasks", []):
                    print(f"  - topic={t.get('topic_id')}, task_id={t.get('task_id')}: {t.get('title')}")
        sys.exit(1)

    resolved_task_id = task.get("task_id")
    project_goal = task.get("description") or task.get("title", "")

    print(f"Person:  {person_name}")
    print(f"Topic:   {topic_name} ({topic_id})")
    print(f"Task:    {task.get('title')} (id={resolved_task_id})")
    print(f"Goal:    {project_goal[:80]}...")

    # Stage 1: Blueprint
    blueprint = stage_blueprint(
        person_data, project_attributes, project_goal,
        topic_name, resolved_task_id, model=blueprint_model,
    )
    if blueprint is None:
        print("Pipeline aborted at Stage 1 (Blueprint).")
        sys.exit(1)

    # Stage 2: Events
    events_data = stage_events(
        person_data, blueprint, project_attributes, model=event_model,
    )
    if events_data is None:
        print("Pipeline aborted at Stage 2 (Events).")
        sys.exit(1)

    # Stage 3: Summaries (per event)
    summary_data = stage_summaries(
        person_data, blueprint, events_data, model=summary_model,
    )
    if summary_data is None:
        print("Pipeline aborted at Stage 3 (Summaries).")
        sys.exit(1)

    # Stage 4: Dialogues
    dialogue_result = stage_dialogues(
        person_data, blueprint, events_data, summary_data,
        dialogue_model=dialogue_model,
        evaluation_model=evaluation_model,
        memory_model=memory_model,
        memory_retrieve_model=memory_retrieve_model,
        dedup_model=dedup_model,
        semantic_schedule_model=semantic_schedule_model,
        max_turns=max_turns,
        max_retries=max_retries,
    )
    if dialogue_result is None:
        print("Pipeline completed with failures in Stage 4 (Dialogues).")
        sys.exit(1)

    # Stage 5: Post-processing
    merged_path = stage_postprocess(
        person_data, blueprint,
        processed_sessions=dialogue_result["processed_sessions"],
    )
    if merged_path is None:
        print("Pipeline completed but Stage 5 (Post-processing) failed.")
        print(f"  Raw dialogues are still available under output/{convert_to_safe_filename(person_name)}/")
    else:
        print(f"  Final dataset: {merged_path}")

    print("\n=== Pipeline Complete ===")
    print(f"  Generated {dialogue_result['total_generated']} dialogue sessions.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dialogues for a persona + topic through the full pipeline."
    )
    parser.add_argument("--name", required=True,
                        help="Persona name (e.g. 'Lin Wanyu', 'Ethan Hunt')")
    parser.add_argument("--topic", required=True,
                        help="Topic ID (e.g. 'fitness', 'travel_planning')")
    parser.add_argument("--task-id", type=str, default=None,
                        help="Specific task ID within the topic (optional)")

    parser.add_argument("--blueprint-model", default=DEFAULT_MODEL)
    parser.add_argument("--event-model", default=DEFAULT_MODEL)
    parser.add_argument("--summary-model", default=DEFAULT_MODEL)
    parser.add_argument("--dialogue-model", default=DEFAULT_MODEL)
    parser.add_argument("--evaluation-model", default=DEFAULT_MODEL)
    parser.add_argument("--memory-model", default=DEFAULT_MODEL)
    parser.add_argument("--memory-retrieve-model", default=DEFAULT_MODEL)
    parser.add_argument("--dedup-model", default=DEFAULT_MODEL)
    parser.add_argument("--semantic-schedule-model", default=DEFAULT_MODEL)

    parser.add_argument("--max-turns", type=int, default=24,
                        help="Max turns per dialogue session (default: 24)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max retries per session on failure (default: 2)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        person_name=args.name,
        topic_id=args.topic,
        task_id=args.task_id,
        blueprint_model=args.blueprint_model,
        event_model=args.event_model,
        summary_model=args.summary_model,
        dialogue_model=args.dialogue_model,
        evaluation_model=args.evaluation_model,
        memory_model=args.memory_model,
        memory_retrieve_model=args.memory_retrieve_model,
        dedup_model=args.dedup_model,
        semantic_schedule_model=args.semantic_schedule_model,
        max_turns=args.max_turns,
        max_retries=args.max_retries,
    )
