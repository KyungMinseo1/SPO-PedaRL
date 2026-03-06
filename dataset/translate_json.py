import json
import time
import argparse
import os
from tqdm import tqdm
from deep_translator import GoogleTranslator

TRANSLATE_FIELDS = [
    "grade", "mission", "question",
    "newHint", "newKnowledgePoint", "newAnalyze"
]

DIALOGUE_FIELDS = [
    "student", "evaluation", "action", "teacher"
]

CHECKPOINT_FILE = "dataset/SocratDataset/translate_checkpoint.json"
translator = GoogleTranslator(source="zh-CN", target="en")


def translate_text_batch(texts: list, retries: int = 3) -> list:
    if not texts:
        return []
    for attempt in range(retries):
        try:
            result = translator.translate_batch(texts)
            time.sleep(0.1)
            if len(result) != len(texts):
                print(f"\n  ⚠️  Batch size mismatch: expected {len(texts)}, got {len(result)}. Retrying...")
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            return result
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"\n  ⚠️  Retrying batch... ({attempt+1}/{retries}), {wait}s wait: {e}")
                time.sleep(wait)
            else:
                print(f"\n  ❌  Batch failed. Using original texts.")
                return texts


def translate_item(item: dict) -> dict:
    translated = dict(item)

    existing_fields = [f for f in TRANSLATE_FIELDS if f in item and isinstance(item[f], str)]
    texts_to_translate = [item[f] for f in existing_fields]

    translated_fields = translate_text_batch(texts_to_translate)

    for i, field in enumerate(existing_fields):
        translated[field] = translated_fields[i]

    dialogue = item.get("dialogue", [])
    new_dialogue = []
    for turn in dialogue:
        existing_turn_fields = [f for f in DIALOGUE_FIELDS if f in turn and isinstance(turn[f], str)]
        turn_texts = [turn[f] for f in existing_turn_fields]

        translated_turn_texts = translate_text_batch(turn_texts)

        new_turn = dict(turn)
        for i, field in enumerate(existing_turn_fields):
            new_turn[field] = translated_turn_texts[i]
        new_dialogue.append(new_turn)

    translated["dialogue"] = new_dialogue
    return translated


def load_checkpoint() -> int:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f).get("last_completed", -1)
    return -1


def save_checkpoint(idx: int):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_completed": idx}, f)


def main():
    parser = argparse.ArgumentParser(description="JSON Chinese->English Translator")
    parser.add_argument("--input",  required=True,  help="Input JSON file path")
    parser.add_argument("--output", required=True,  help="Output JSON file path")
    parser.add_argument("--resume", action="store_true", help="Resume from interrupted point")
    parser.add_argument("--start",  type=int, default=0, help="Start index for translation")
    parser.add_argument("--limit",  type=int, default=None, help="Test mode: translate only first N items")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]
        print(f"Test mode: translating only first {args.limit} items.")

    total = len(data)
    print(f"Total items: {total}")
    start_idx = args.start
    results = []

    if args.resume and os.path.exists(args.output):
        last = load_checkpoint()
        if last >= 0:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            start_idx = last + 1
            print(f"Resuming from item {start_idx}.")

    print(f"\nStarting translation...\n")
    try:
        for i in tqdm(range(start_idx, total), initial=start_idx, total=total, unit="items"):
            translated = translate_item(data[i])
            results.append(translated)

            if (i + 1) % 50 == 0 or i == total - 1:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                save_checkpoint(i)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Resume later with --resume flag.")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        save_checkpoint(len(results) - 1)
        return

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\nDone! Saved to: {args.output} ({len(results)} items)")


if __name__ == "__main__":
    main()