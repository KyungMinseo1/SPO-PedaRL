import json
import time
import argparse
import os
from tqdm import tqdm
from deep_translator import GoogleTranslator

TRANSLATE_FIELDS = {
    "grade", "mission", "question",
    "newHint", "newKnowledgePoint", "newAnalyze"
}

DIALOGUE_FIELDS = {
    "student", "evaluation", "action", "teacher"
}

CHECKPOINT_FILE = "translate_checkpoint.json"
translator = GoogleTranslator(source="zh-CN", target="en")


def translate_text_batch(texts: list, retries: int = 3) -> list:
    if not isinstance(texts, list):
        texts = [texts]
    results = []
    for attempt in range(retries):
        try:
            translated_list = translator.translate_batch(texts)
            time.sleep(0.1)
            return translated_list
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"\n  ⚠️  Retrying batch ... ({attempt+1}/{retries}), {wait}s wait: {e}")
                time.sleep(wait)
            else:
                print(f"\n  ❌  Batch Translate Failed. Using original texts.")
    return results

def translate_text(text: str, retries: int = 3) -> str:
    if not text or not isinstance(text, str) or not text.strip():
        return text
    for attempt in range(retries):
        try:
            result = translator.translate(text)
            time.sleep(0.1)
            return result
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"\n  ⚠️  Retrying ... ({attempt+1}/{retries}), {wait}s wait: {e}")
                time.sleep(wait)
            else:
                print(f"\n  ❌  Translate Failed. Using original text: {text[:50]}...")
                return text


def translate_item(item: dict) -> dict:
    translated = dict(item)

    translate_list = [item.get(key) for key in TRANSLATE_FIELDS]
    dialouge = item.get("dialogue", [])
    dialogue_list = []
    for turn in dialouge:
        turn_list = []
        for field in DIALOGUE_FIELDS:
            if field in turn:
                turn_list.append(turn.get(field))
        dialogue_list.append(turn_list)

    translated_fields = translate_text_batch(translate_list)
    translated_dialogue = []
    for turn in dialogue_list:
        translated_turn = translate_text_batch(turn)
        translated_dialogue.append(translated_turn)

    # Update the translated item with the translated fields and dialogue
    for i, field in enumerate(TRANSLATE_FIELDS):
        if field in translated:
            translated[field] = translated_fields[i]
    for i, turn in enumerate(translated_dialogue):
        for j, field in enumerate(DIALOGUE_FIELDS):
            translated["dialogue"][i][field] = turn[j]

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
    parser = argparse.ArgumentParser(description="JSON Chinese→English Translator")
    parser.add_argument("--input",  required=True,  help="Input JSON file path")
    parser.add_argument("--output", required=True,  help="Output JSON file path")
    parser.add_argument("--resume", action="store_true", help="Resume from interrupted point")
    parser.add_argument("--start",  type=int, default=0, help="Start index for translation")
    parser.add_argument("--limit",  type=int, default=None, help="Test mode: translate only first N items")
    args = parser.parse_args()

    print(f"📂 Loading: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]
        print(f"🔬 Test mode: translating only first {args.limit} items.")

    total = len(data)
    print(f"📊 Total items: {total}")
    start_idx = args.start
    results = []

    if args.resume and os.path.exists(args.output):
        last = load_checkpoint()
        if last >= 0:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            start_idx = last + 1
            print(f"⏩ Starting from item {start_idx}.")
    else:
        results = []

    print(f"\n🚀 Starting translation...\n")
    try:
        for i in tqdm(range(start_idx, total), initial=start_idx, total=total, unit="items"):
            translated = translate_item(data[i])
            results.append(translated)

            if (i + 1) % 50 == 0 or i == total - 1:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                save_checkpoint(i)

    except KeyboardInterrupt:
        print("\n\n⏸️  Translation interrupted. You can resume later with the --resume flag.")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        save_checkpoint(len(results) - 1)
        return

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\n✅ Completed! Results saved to: {args.output}")
    print(f"   Total {len(results)} items translated")


if __name__ == "__main__":
    main()