"""
Enriches retrieval results by adding session_id to each entity.
This enables the compute_auto_metrics script to properly evaluate retrieval.

The script searches dialogue content to find which session(s) contain
content related to each retrieved entity.
"""

import json
import argparse
import os
import re
from collections import defaultdict


def load_dialogues(dialogues_file):
    """Load dialogues and create searchable index."""
    with open(dialogues_file, 'r') as f:
        data = json.load(f)

    # Build session content index: session_id -> concatenated text
    session_content = {}
    session_ids = []

    for dialogue in data['dialogues']:
        sid = dialogue.get('session_identifier')
        if not sid:
            continue
        session_ids.append(sid)

        # Concatenate all turn content for this session
        content_parts = []
        for turn in dialogue.get('dialogue_turns', []):
            text = turn.get('content', '')
            if text:
                content_parts.append(text.lower())

        session_content[sid] = ' '.join(content_parts)

    return session_content, session_ids


def extract_keywords(text):
    """Extract meaningful keywords from entity content."""
    # Remove quotes and clean up
    text = text.strip('"\'')
    text = text.lower()

    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                  'from', 'as', 'into', 'through', 'during', 'before', 'after',
                  'above', 'below', 'between', 'under', 'again', 'further',
                  'then', 'once', 'here', 'there', 'when', 'where', 'why',
                  'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                  'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                  'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                  'because', 'until', 'while', 'this', 'that', 'these', 'those',
                  'their', 'they', 'them', 'its', 'user', 'users', 'plans',
                  'part', 'per', 'week', 'based', 'using', 'used', 'use'}

    # Extract words
    words = re.findall(r'\b[a-z]+\b', text)
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    return keywords


def find_matching_sessions(entity_content, entity_name, session_content, max_matches=5):
    """Find sessions that contain content related to the entity."""
    # Clean entity content and name
    content = entity_content.strip('"\'').lower()
    name = entity_name.strip('"\'').lower().replace('_', ' ')

    # Extract keywords from both content and name
    content_keywords = extract_keywords(content)
    name_keywords = extract_keywords(name)
    all_keywords = list(set(content_keywords + name_keywords))

    if not all_keywords:
        return []

    # Score each session by keyword matches
    session_scores = []
    for sid, session_text in session_content.items():
        score = 0
        matched_keywords = []

        for keyword in all_keywords:
            if keyword in session_text:
                score += 1
                matched_keywords.append(keyword)

        # Bonus for matching multiple keywords together (phrase-like)
        if len(matched_keywords) >= 2:
            score += len(matched_keywords) * 0.5

        if score > 0:
            session_scores.append((sid, score, matched_keywords))

    # Sort by score descending
    session_scores.sort(key=lambda x: -x[1])

    # Return top matches
    return [sid for sid, score, _ in session_scores[:max_matches]]


def enrich_retrieval_results(retrieval_file, dialogues_file, output_file):
    """Enrich retrieval results with session IDs."""
    print(f"Loading dialogues from {dialogues_file}...")
    session_content, all_session_ids = load_dialogues(dialogues_file)
    print(f"  Loaded {len(all_session_ids)} sessions")

    print(f"Loading retrieval results from {retrieval_file}...")
    with open(retrieval_file, 'r') as f:
        retrieval_results = json.load(f)

    total_items = 0
    enriched_items = 0

    for query_id, result_obj in retrieval_results.items():
        ranked_items = result_obj.get('ranked_items', [])

        for item in ranked_items:
            total_items += 1
            res_type = item.get('res_type')

            if res_type == 'entity':
                entity_name = item.get('entity_name', '')
                entity_content = item.get('content', '')

                # Find matching sessions
                matching_sessions = find_matching_sessions(
                    entity_content, entity_name, session_content
                )

                if matching_sessions:
                    # Add the best matching session as entity_id
                    item['entity_id'] = matching_sessions[0]
                    # Also store all matches for reference
                    item['matched_session_ids'] = matching_sessions
                    enriched_items += 1

            elif res_type == 'chunk':
                # Chunks should already have chunk_id
                if item.get('chunk_id'):
                    enriched_items += 1

    print(f"  Enriched {enriched_items}/{total_items} items with session IDs")

    # Save enriched results
    with open(output_file, 'w') as f:
        json.dump(retrieval_results, f, indent=2)
    print(f"  Saved to {output_file}")

    return retrieval_results


def process_all(retrieval_result_dir, input_data_dir):
    """Process all retrieval result files."""
    if not os.path.exists(retrieval_result_dir):
        print(f"Error: Retrieval result directory not found at {retrieval_result_dir}")
        return

    if not os.path.exists(input_data_dir):
        print(f"Error: Input data directory not found at {input_data_dir}")
        return

    # Find all subdirectories with example_retrieval_results.json
    subdirs = []
    for item in os.listdir(retrieval_result_dir):
        item_path = os.path.join(retrieval_result_dir, item)
        if os.path.isdir(item_path):
            retrieval_file = os.path.join(item_path, 'example_retrieval_results.json')
            if os.path.exists(retrieval_file):
                subdirs.append((item, retrieval_file))

    subdirs.sort()
    print(f"Found {len(subdirs)} retrieval result files to process")

    for name, retrieval_file in subdirs:
        # Construct dialogues file path
        dialogues_file = os.path.join(input_data_dir, f"{name}_dialogues_256k.json")

        # Output to enriched file
        output_file = os.path.join(os.path.dirname(retrieval_file),
                                   'example_retrieval_results_enriched.json')

        if not os.path.exists(dialogues_file):
            print(f"Skipping {name}: Dialogues file not found at {dialogues_file}")
            continue

        print(f"\nProcessing {name}...")
        try:
            enrich_retrieval_results(retrieval_file, dialogues_file, output_file)
            print(f"Done {name}")
        except Exception as e:
            print(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Enrich retrieval results with session IDs for metrics evaluation'
    )
    parser.add_argument('--retrieval_file', type=str,
                        help='Single retrieval results file to process')
    parser.add_argument('--dialogues_file', type=str,
                        help='Dialogues file for the retrieval results')
    parser.add_argument('--output_file', type=str,
                        help='Output file for enriched results')
    parser.add_argument('--process_all', action='store_true',
                        help='Process all retrieval result files')
    parser.add_argument('--retrieval_result_dir', type=str,
                        default='eval/retrieval_result',
                        help='Directory containing retrieval result subdirectories')
    parser.add_argument('--input_data_dir', type=str,
                        default='dataset',
                        help='Directory containing input dialogue files')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.process_all:
        process_all(args.retrieval_result_dir, args.input_data_dir)
    elif args.retrieval_file and args.dialogues_file:
        output_file = args.output_file or args.retrieval_file.replace('.json', '_enriched.json')
        enrich_retrieval_results(args.retrieval_file, args.dialogues_file, output_file)
    else:
        print("Usage:")
        print("  Process all: python enrich_retrieval_with_session_ids.py --process_all")
        print("  Single file: python enrich_retrieval_with_session_ids.py --retrieval_file FILE --dialogues_file FILE")
