import re
import pandas as pd
import random

def dictionary_generator(text):
    """
    Parses a raw transcript string into a list of role/content dictionaries.
    """
    if not isinstance(text, str):
        return []

    messages = []
    # Split the text dynamically keeping the delimiters
    parts = re.split(r"(User:|Assistant:)", text)
    current_role = None

    for chunk in parts:
        chunk = chunk.strip()
        if not chunk:
            continue

        if chunk == "User:":
            current_role = "user"
        elif chunk == "Assistant:":
            current_role = "assistant"
        elif current_role:
            messages.append({"role": current_role, "content": chunk})

    return messages

def get_truncated_prompts(csv_path):
    """
    Loads a dataset and truncates multi-turn transcripts dynamically.
    Ensures truncation happens in the >50% length range and always ends
    with a User turn, primed for an Assistant response.
    """
    df = pd.read_csv(csv_path)
    prompts = []

    for transcript in df['transcript'].dropna():
        parts = re.split(r'(User:|Assistant:)', str(transcript))

        turns = []
        for i in range(1, len(parts), 2):
            role = parts[i].replace(":", "").strip()
            content = parts[i+1].strip()
            turns.append({"role": role, "content": content})

        user_indices = [i for i, t in enumerate(turns) if t['role'] == 'User']
        if not user_indices:
            continue

        # Calculate the 50% threshold
        min_user_idx = len(user_indices) // 2

        # Pick a random User turn from the second half of the conversation
        chosen_turn_idx = random.choice(user_indices[min_user_idx:])

        # Reconstruct the string up to this chosen User turn
        truncated_history = turns[:chosen_turn_idx + 1]

        prompt_text = ""
        for turn in truncated_history:
            prompt_text += f"{turn['role']}: {turn['content']}\n"

        # Append 'Assistant:' to prime the model for generation
        prompt_text += "Assistant: "
        prompts.append(prompt_text)

    return prompts
