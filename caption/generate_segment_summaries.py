import json
import os
from pathlib import Path
from openai import OpenAI
import argparse


# Define paths
SEGMENT_FILE = '/root/autodl-tmp/VideoTree/data/nextqa_segment_sbd.json'
CAPTION_DIR = '/root/autodl-tmp/VideoTree/data/per_second_category_captions/'
DURATIONS_FILE = '/root/autodl-tmp/VideoTree/data/nextqa/durations.json'
OUTPUT_FILE = '/root/autodl-tmp/VideoTree/data/segment_summaries.jsonl'
ERROR_LOG = '/root/autodl-tmp/VideoTree/data/segment_summary_errors.jsonl'
PERMANENT_FAILURES = '/root/autodl-tmp/VideoTree/data/permanent_failures.json'

# Ensure output directory exists
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

# Question type definitions
QUESTION_TYPES = {
    'causal': 'Causal questions - Focus on reasons (why) and methods (how) of actions',
    'temporal': 'Temporal questions - Focus on action sequences (before/after/current)',
    'descriptive': 'Descriptive questions - Focus on scene information (location/objects/counting)'
}

# Individual prompts for each category
CAUSAL_PROMPT = """Based on the following video captions (captured every 2 seconds), provide a comprehensive summary focusing on CAUSAL RELATIONSHIPS:
- WHY actions occur (reasons, motivations, causes)
- HOW actions are performed (methods, techniques, processes)

Video Information:
- Total video duration: {total_duration}s
- Current segment: {start_time}s to {end_time}s 
- This segment is part of the complete video

Captions:
{captions}

IMPORTANT REQUIREMENTS:
- Base your summary ONLY on factual information present in the captions above
- DO NOT invent, assume, or hallucinate any information not explicitly mentioned
- DO NOT repeat the same information multiple times
- Provide a concise summary with unique, non-redundant information
- Focus specifically on causal aspects (why and how)

Summary:"""

TEMPORAL_PROMPT = """Based on the following video captions (captured every 2 seconds), provide a comprehensive summary focusing on TEMPORAL SEQUENCES:
- What happens BEFORE key actions
- What is happening DURING the segment
- What happens AFTER certain events
- The order and timing of events

Video Information:
- Total video duration: {total_duration}s
- Current segment: {start_time}s to {end_time}s 
- This segment is part of the complete video

Captions:
{captions}

IMPORTANT REQUIREMENTS:
- Base your summary ONLY on factual information present in the captions above
- DO NOT invent, assume, or hallucinate any information not explicitly mentioned
- DO NOT repeat the same information multiple times
- Provide a concise summary with unique, non-redundant information
- Focus specifically on temporal sequences and the order of events

Summary:"""

DESCRIPTIVE_PROMPT = """Based on the following video captions (captured every 2 seconds), provide a comprehensive summary focusing on DESCRIPTIVE INFORMATION:
- WHERE the scene takes place (location, setting)
- WHAT objects are present (items, props, environment)
- WHO is in the scene (people, animals, their attributes)
- HOW MANY of each element (counting information)

Video Information:
- Total video duration: {total_duration}s
- Current segment: {start_time}s to {end_time}s 
- This segment is part of the complete video

Captions:
{captions}

IMPORTANT REQUIREMENTS:
- Base your summary ONLY on factual information present in the captions above
- DO NOT invent, assume, or hallucinate any information not explicitly mentioned
- DO NOT repeat the same information multiple times
- Provide a concise summary with unique, non-redundant information
- Focus specifically on observable, descriptive details

Summary:"""


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(data, filepath):
    """Append data to JSONL file"""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def get_processed_video_ids(filepath):
    """Get set of video IDs that have already been processed"""
    processed = set()
    if not Path(filepath).exists():
        return processed

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'video_id' in data:
                        processed.add(data['video_id'])
    except Exception as e:
        print(f"Warning: Error reading processed videos: {e}")

    return processed


def check_incomplete_videos(filepath):
    """
    Check for videos with incomplete summaries (any None values)
    Returns dict mapping video_id to list of (segment_idx, category) tuples that need regeneration
    """
    incomplete = {}
    if not Path(filepath).exists():
        return incomplete

    # Load permanent failures to skip them
    permanent_failures = load_permanent_failures()

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    video_id = data.get('video_id')
                    segments = data.get('segments', [])

                    incomplete_items = []
                    for seg_idx, segment in enumerate(segments):
                        summaries = segment.get('summaries', {})
                        for category, summary in summaries.items():
                            if summary is None:
                                # Check if this is a permanent failure
                                failure_key = f"{video_id}_{seg_idx}_{category}"
                                if failure_key not in permanent_failures:
                                    incomplete_items.append((seg_idx, category))

                    if incomplete_items:
                        incomplete[video_id] = incomplete_items
    except Exception as e:
        print(f"Warning: Error checking incomplete videos: {e}")

    return incomplete


def load_permanent_failures():
    """Load set of permanent failure keys that should not be retried"""
    if not Path(PERMANENT_FAILURES).exists():
        return set()

    try:
        with open(PERMANENT_FAILURES, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('failures', []))
    except Exception as e:
        print(f"Warning: Error loading permanent failures: {e}")
        return set()


def save_permanent_failure(video_id, segment_idx, category, reason):
    """Record a permanent failure (like content filter) that should not be retried"""
    failure_key = f"{video_id}_{segment_idx}_{category}"

    failures = load_permanent_failures()
    failures.add(failure_key)

    try:
        failure_data = {
            'failures': list(failures),
            'details': {}
        }

        # Load existing details if file exists
        if Path(PERMANENT_FAILURES).exists():
            with open(PERMANENT_FAILURES, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                failure_data['details'] = existing.get('details', {})

        # Add this failure's details
        failure_data['details'][failure_key] = {
            'video_id': video_id,
            'segment_idx': segment_idx,
            'category': category,
            'reason': reason,
            'timestamp': pd.Timestamp.now().isoformat() if 'pd' in dir() else None
        }

        with open(PERMANENT_FAILURES, 'w', encoding='utf-8') as f:
            json.dump(failure_data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"Warning: Could not save permanent failure: {e}")


def reprocess_incomplete_segment(video_id, segment_idx, categories_to_regenerate,
                                   segment_data, caption_data, total_duration):
    """
    Regenerate summaries for specific categories in a segment

    Returns:
        Dictionary with updated summaries
    """
    segment = segment_data[video_id][segment_idx]
    uid = segment['uid']
    start_time = segment['start_time']
    end_time = segment['end_time']
    duration = segment['duration']

    print(f"  Regenerating segment {segment_idx + 1} (uid: {uid}), categories: {categories_to_regenerate}")

    # Initialize summaries dict if not exists
    summaries = {}

    for category in categories_to_regenerate:
        captions_text = get_segment_captions(caption_data, start_time, end_time, category, video_id)

        if "No captions available" in captions_text:
            print(f"    ⚠️  No captions for {category}")
            summaries[category] = None
            continue

        summary = generate_summary(captions_text, category, start_time, end_time, duration, total_duration)

        if summary:
            summaries[category] = summary
            print(f"    ✓ {category} regenerated")
        else:
            summaries[category] = None
            print(f"    ✗ {category} failed again")

    return summaries


def get_segment_captions(caption_data, start_time, end_time, category, video_id):
    """
    Extract captions for a specific time segment and category

    Args:
        caption_data: Dictionary with structure {video_id: {timestamp: {category: caption}}}
        start_time: Segment start time in seconds
        end_time: Segment end time in seconds
        category: One of 'causal', 'temporal', 'descriptive'
        video_id: Video identifier

    Returns:
        String with formatted captions
    """
    captions = []

    # Get the video-specific captions
    video_captions = caption_data.get(video_id, {})

    # Iterate through timestamps
    for timestamp_str, categories_dict in video_captions.items():
        try:
            timestamp = float(timestamp_str)
            # Check if timestamp is within segment range
            if start_time <= timestamp <= end_time:
                # Get caption for the specific category
                if isinstance(categories_dict, dict) and category in categories_dict:
                    caption_text = categories_dict[category]
                    captions.append(f"[{timestamp}s]: {caption_text}")
        except (ValueError, TypeError) as e:
            continue

    if not captions:
        return "No captions available for this segment and category."

    return '\n'.join(captions)


def generate_summary(captions_text, category, start_time, end_time, duration, total_duration, retry_simplified=False):
    """
    Use LLM to generate a summary for a segment

    Args:
        captions_text: Formatted caption text
        category: Question type category
        start_time: Segment start time
        end_time: Segment end time
        duration: Segment duration
        total_duration: Total video duration
        retry_simplified: If True, use simplified prompt to avoid content filter

    Returns:
        Generated summary text or None if error
    """
    # Select prompt based on category using if-elif-else
    if category == 'causal':
        prompt_template = CAUSAL_PROMPT
    elif category == 'temporal':
        prompt_template = TEMPORAL_PROMPT
    elif category == 'descriptive':
        prompt_template = DESCRIPTIVE_PROMPT
    else:
        print(f"    ❌ Unknown category: {category}")
        return None

    # Use simplified prompt if retry_simplified is True
    if retry_simplified:
        # Simplified prompt without detailed instructions to avoid content filter
        prompt = f"""Summarize the following video segment focusing on {category} aspects.

Video: {start_time}s to {end_time}s (total: {total_duration}s)

Captions:
{captions_text}

Provide a brief, factual summary:"""
    else:
        # Format the prompt with segment information
        prompt = prompt_template.format(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            total_duration=total_duration,
            captions=captions_text
        )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a video content analyst. Provide factual summaries based only on the given captions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )

        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content.strip()

        return None

    except Exception as e:
        error_str = str(e)

        # Check if it's a content filter error
        if 'content_filter' in error_str or 'content management policy' in error_str:
            if not retry_simplified:
                print(f"    ⚠️  Content filter triggered, retrying with simplified prompt...")
                # Retry with simplified prompt
                return generate_summary(captions_text, category, start_time, end_time,
                                       duration, total_duration, retry_simplified=True)
            else:
                print(f"    ❌ Content filter error (even with simplified prompt): {error_str[:100]}")
                return None
        else:
            print(f"    ❌ Error generating summary: {e}")
            return None


def process_video(video_id, segments, caption_data, total_duration):
    """
    Process all segments for a single video

    Args:
        video_id: Video identifier
        segments: List of segment dictionaries
        caption_data: Caption data for the video
        total_duration: Total video duration in seconds

    Returns:
        Dictionary with video_id and segment results
    """
    video_results = []

    for idx, segment in enumerate(segments, 1):
        uid = segment['uid']
        start_time = segment['start_time']
        end_time = segment['end_time']
        duration = segment['duration']

        print(f"  [{idx}/{len(segments)}] Segment {uid}: {start_time}s - {end_time}s ({duration}s)")

        segment_result = {
            'uid': uid,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'summaries': {}
        }

        # Get available categories from the first timestamp in this video
        video_captions = caption_data.get(video_id, {})
        if video_captions:
            first_timestamp = list(video_captions.values())[0]
            available_categories = list(first_timestamp.keys()) if isinstance(first_timestamp, dict) else []
        else:
            available_categories = []

        # Generate summary for each available category
        for category in available_categories:
            captions_text = get_segment_captions(caption_data, start_time, end_time, category, video_id)

            if "No captions available" in captions_text:
                print(f"    ⚠️  No captions for {category}")
                segment_result['summaries'][category] = None
                continue

            summary = generate_summary(captions_text, category, start_time, end_time, duration, total_duration)

            if summary:
                segment_result['summaries'][category] = summary
                print(f"    ✓ {category}")
            else:
                segment_result['summaries'][category] = None
                print(f"    ✗ {category} (failed)")

        video_results.append(segment_result)

    return {
        'video_id': video_id,
        'segments': video_results
    }


def main(start_idx=None, end_idx=None, api_key=None):
    """
    Main processing function

    Args:
        start_idx: Starting video index (optional)
        end_idx: Ending video index (optional)
        api_key: OpenAI API key (optional, uses env var if not provided)
    """
    # Set API key if provided
    if api_key:
        global client
        client = OpenAI(api_key=api_key, base_url='https://opus.gptuu.com/v1')

    # Load segment data
    print(f"Loading segment data from {SEGMENT_FILE}...")
    segment_data = load_json(SEGMENT_FILE)

    # Load video durations
    print(f"Loading video durations from {DURATIONS_FILE}...")
    durations_data = load_json(DURATIONS_FILE)
    print(f"Loaded durations for {len(durations_data)} videos")

    video_ids = list(segment_data.keys())
    total_videos = len(video_ids)

    # Apply range if specified
    if start_idx is not None and end_idx is not None:
        video_ids = video_ids[start_idx:end_idx]
        print(f"Processing videos {start_idx} to {end_idx} (out of {total_videos})")
    else:
        print(f"Processing all {total_videos} videos")

    processed_count = 0
    error_count = 0
    skipped_count = 0
    regenerated_count = 0

    # Check for incomplete videos and regenerate them
    print(f"\n{'='*60}")
    print("Checking for incomplete videos...")
    incomplete_videos = check_incomplete_videos(OUTPUT_FILE)

    if incomplete_videos:
        print(f"Found {len(incomplete_videos)} videos with incomplete summaries")

        # Read all existing results into memory
        all_results = {}
        if Path(OUTPUT_FILE).exists():
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        video_id = data.get('video_id')
                        if video_id:
                            all_results[video_id] = data

        # Reprocess incomplete videos
        for video_id, incomplete_items in incomplete_videos.items():
            if video_id not in segment_data:
                print(f"  ⚠️  Video {video_id} not found in segment data, skipping...")
                continue

            print(f"\n{'='*60}")
            print(f"Reprocessing incomplete video: {video_id}")
            print(f"  Incomplete items: {len(incomplete_items)}")

            # Get video info
            total_duration = durations_data.get(video_id, 0)
            caption_file = Path(CAPTION_DIR) / f"{video_id}.json"

            if not caption_file.exists():
                print(f"  ⏭️  Caption file not found, keeping original...")
                continue

            try:
                caption_data = load_json(caption_file)

                # Group incomplete items by segment
                segments_to_fix = {}
                for seg_idx, category in incomplete_items:
                    if seg_idx not in segments_to_fix:
                        segments_to_fix[seg_idx] = []
                    segments_to_fix[seg_idx].append(category)

                # Get existing result from memory
                existing_result = all_results.get(video_id)

                if existing_result:
                    # Update incomplete segments
                    for seg_idx, categories in segments_to_fix.items():
                        updated_summaries = reprocess_incomplete_segment(
                            video_id, seg_idx, categories,
                            segment_data, caption_data, total_duration
                        )
                        
                        # Ensure the segments list has enough elements
                        if seg_idx >= len(existing_result.get('segments', [])):
                            print(f"  ⚠️  Segment index {seg_idx} out of range, skipping...")
                            continue
                        
                        # Ensure the segment has a 'summaries' dictionary
                        if 'summaries' not in existing_result['segments'][seg_idx]:
                            existing_result['segments'][seg_idx]['summaries'] = {}
                        
                        # Update only the regenerated categories in the segment's summaries
                        existing_result['segments'][seg_idx]['summaries'].update(updated_summaries)

                    # Update the in-memory dict (will be written back later)
                    all_results[video_id] = existing_result
                    regenerated_count += 1
                    print(f"  ✓ Regenerated incomplete summaries for {video_id}")

            except Exception as e:
                print(f"  ❌ Error reprocessing video: {e}")
                continue

        # Rewrite the entire output file with updated results (safely)
        if regenerated_count > 0:
            print(f"\n{'='*60}")
            print(f"Updating output file with {regenerated_count} regenerated videos...")

            # Write to temporary file first (safe operation)
            temp_file = Path(OUTPUT_FILE).with_suffix('.tmp')
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    for video_id, result in all_results.items():
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

                # Backup original file
                backup_file = Path(OUTPUT_FILE).with_suffix('.backup')
                if Path(OUTPUT_FILE).exists():
                    import shutil
                    shutil.copy2(OUTPUT_FILE, backup_file)
                    print(f"  ✓ Created backup: {backup_file}")

                # Atomic rename (safe operation)
                temp_file.replace(OUTPUT_FILE)
                print("  ✓ Output file updated (replaced existing entries)")

                # Remove backup after successful update
                if backup_file.exists():
                    backup_file.unlink()
                    print("  ✓ Backup removed (update successful)")

            except Exception as e:
                print(f"  ❌ Error updating file: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                print(f"  ℹ️  Original file unchanged")
                raise
    else:
        print("✓ No incomplete videos found")

    print(f"\n{'='*60}")
    print("Starting normal processing...")

    # Load already processed video IDs
    processed_videos = get_processed_video_ids(OUTPUT_FILE)
    if processed_videos:
        print(f"Found {len(processed_videos)} already processed videos, will skip them")

    for video_id in video_ids:
        segments = segment_data[video_id]
        print(f"\n{'='*60}")
        print(f"Processing video: {video_id}")

        # Check if already processed
        if video_id in processed_videos:
            print(f"  ⏭️  Already processed, skipping...")
            skipped_count += 1
            continue

        print(f"  Segments: {len(segments)}")

        # Get video total duration
        total_duration = durations_data.get(video_id)
        if total_duration is None:
            print(f"  ⚠️  Warning: Duration not found for video {video_id}, using default 0")
            total_duration = 0
        else:
            print(f"  Total duration: {total_duration}s")

        # Load caption file
        caption_file = Path(CAPTION_DIR) / f"{video_id}.json"

        if not caption_file.exists():
            print(f"  ⏭️  Caption file not found, skipping...")
            save_jsonl({
                'video_id': video_id,
                'error': 'Caption file not found',
                'file': str(caption_file)
            }, ERROR_LOG)
            skipped_count += 1
            continue

        try:
            caption_data = load_json(caption_file)
            # Get the actual timestamps for this video
            video_captions = caption_data.get(video_id, {})
            print(f"  ✓ Loaded {len(video_captions)} caption timestamps")

            # Process all segments for this video
            video_result = process_video(video_id, segments, caption_data, total_duration)

            # Save to JSONL file (append mode)
            save_jsonl(video_result, OUTPUT_FILE)

            processed_count += 1
            print(f"  ✓ Saved to {OUTPUT_FILE}")

        except Exception as e:
            print(f"  ❌ Error processing video: {e}")
            save_jsonl({
                'video_id': video_id,
                'error': str(e)
            }, ERROR_LOG)
            error_count += 1
            continue

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  🔄 Regenerated: {regenerated_count} videos")
    print(f"  ✓ Successfully processed: {processed_count} videos")
    print(f"  ⏭️  Skipped: {skipped_count} videos")
    print(f"  ✗ Errors: {error_count} videos")
    print(f"  📁 Results saved to: {OUTPUT_FILE}")
    if error_count > 0:
        print(f"  📁 Error log: {ERROR_LOG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate category-specific summaries for video segments')
    parser.add_argument('--start', type=int, help='Start index for video processing')
    parser.add_argument('--end', type=int, help='End index for video processing')
    parser.add_argument('--api-key', type=str, help='OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)')

    args = parser.parse_args()
    main(args.start, args.end, args.api_key)

