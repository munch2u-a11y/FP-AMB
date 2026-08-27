import re
from typing import List, Tuple

GREETINGS_AND_FILLERS = {
    "hey", "hi", "hello", "wow", "thanks", "thank", "sorry", "yup", "yeah", 
    "woohoo", "guess", "what", "really", "awesome", "fantastic", "cool", "nice",
    "sounds", "looks", "gonna", "gotta", "wanna", "doing", "think", "just", 
    "also", "very", "much", "well", "good", "great", "loved", "liked", "hope",
    "doing", "talked", "spoke", "tell", "telling", "think", "know"
}

def extract_topical_entities(text: str) -> Tuple[str, List[str]]:
    """Extracts speaker name (as factor/metadata) and a list of Topical Subject Nouns
    (e.g., Speaker: 'Caroline', Topics: ['Adoption', 'Counseling', 'Interviews']).
    """
    clean_text = text.strip()
    speaker = "User"
    
    if ":" in clean_text:
        parts = clean_text.split(":", 1)
        speaker = parts[0].strip().title()
        clean_text = parts[1].strip()

    clean_text = re.sub(r'\[.*?\]', '', clean_text).strip()

    # Find Capitalized Subject Proper Nouns & Noun Phrases
    words = re.findall(r'\b[A-Z][a-zA-Z0-9_\-]+\b', clean_text)
    topics: List[str] = []

    for word in words:
        w_lower = word.lower()
        if w_lower not in GREETINGS_AND_FILLERS and len(word) > 2 and w_lower != speaker.lower():
            if word not in topics:
                topics.append(word)

    # Fallback: Find key noun-like words (>3 chars) if no proper nouns found
    if not topics:
        all_words = re.findall(r'\b[a-zA-Z0-9_\-]+\b', clean_text)
        for word in all_words:
            w_lower = word.lower()
            if w_lower not in GREETINGS_AND_FILLERS and len(word) > 3 and not word.isdigit() and w_lower != speaker.lower():
                w_title = word.title()
                if w_title not in topics:
                    topics.append(w_title)
                    if len(topics) >= 2:
                        break

    if not topics:
        topics = ["General"]

    return speaker, topics
