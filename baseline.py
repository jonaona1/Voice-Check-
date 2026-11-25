"""
VoiceCheck - Week 2 NON-AI baseline (voice command classifier)
--------------------------------------------------------------
Input : phrases.csv   (columns: text, label)
Output: output/clean_phrases.csv, output/predictions.csv, report in the terminal

No machine learning. A phrase is turned into one of the commands
Register / Repeat / Cancel / Help (or Unknown) using keyword rules,
so every answer can be checked by hand.

Run:  python baseline.py
"""
import csv
import os
import re
from collections import Counter

from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "phrases.csv")
OUT_DIR = os.path.join(HERE, "output")

COMMANDS = ("Register", "Repeat", "Cancel", "Help")

# ------------------------------------------------------------------
# Keyword rules
# ------------------------------------------------------------------
KEYWORDS = {
    "Register": ["register", "check in", "check me in", "sign in", "sign me in", "mark me",
                 "mark my attendance", "mark present", "present", "attendance", "scan",
                 "here", "in class", "do it", "go ahead", "start", "begin", "get started",
                 "ready", "yes", "yeah", "ok", "okay", "confirm", "submit", "record", "log"],
    "Repeat":   ["repeat", "again", "once more", "one more time", "what did you",
                 "what was that", "did not hear", "did not catch", "did not get",
                 "missed that", "come again", "sorry what", "pardon", "say that", "say it"],
    "Cancel":   ["cancel", "stop", "never mind", "nevermind", "forget", "abort", "quit",
                 "exit", "go back", "back", "no", "not now", "close", "leave it", "undo",
                 "start over", "changed my mind", "wrong", "turn off", "turn it off",
                 "do not want"],
    "Help":     ["help", "what can i", "what are", "what should i", "what do i", "what now",
                 "options", "commands", "how does", "how do i", "lost", "confused", "stuck",
                 "do not understand", "dont understand", "do not know", "instructions",
                 "explain", "guide me", "tell me what to do", "where", "what can you do",
                 "show me what to say"],
}
NEGATIONS = ["dont", "do not", "not"]


def normalize_text(text):
    """lowercase, remove apostrophes, keep only letters/digits/spaces, collapse spaces.
    "Don't  Register!" -> "dont register" """
    if text is None:
        return ""
    text = text.lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _has(text, phrase):
    """True if `phrase` appears as whole words inside `text`."""
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def classify_command(phrase):
    """Returns (command, explanation). Rules, in order:
    1. empty phrase                                          -> Unknown
    2. negation or Cancel word together with a Register word -> Cancel
    3. count keyword hits per command; the single highest    -> that command
    4. no hits, or a tie for first place                     -> Unknown (ask again)"""
    text = normalize_text(phrase)
    if not text:
        return "Unknown", "empty phrase"

    reg_hit = any(_has(text, k) for k in KEYWORDS["Register"])
    stop_hit = [w for w in NEGATIONS + KEYWORDS["Cancel"] if _has(text, w)]
    if stop_hit and reg_hit:
        return "Cancel", f"'{stop_hit[0]}' + register word"

    scores = {c: sum(_has(text, k) for k in KEYWORDS[c]) for c in COMMANDS}
    best = max(scores.values())
    if best == 0:
        return "Unknown", "no keyword matched"
    winners = [c for c in COMMANDS if scores[c] == best]
    if len(winners) > 1:
        return "Unknown", "tie: " + " = ".join(winners)
    return winners[0], f"{best} keyword hit(s)"


# ------------------------------------------------------------------
# Preprocessing of the CSV
# ------------------------------------------------------------------
def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def preprocess(rows):
    """Cleans the raw rows. Returns (clean_rows, dropped_rows).
    - text : normalized (see normalize_text); empty -> dropped
    - label: trimmed, first letter capital; empty or not one of the 4 commands -> dropped
    - exact duplicates (same text + same label) -> keep the first one
    - conflicts (same text, different labels)   -> drop ALL copies (we can't know which is right)"""
    dropped, candidates = [], []
    for i, r in enumerate(rows, start=2):                       # row 1 is the header
        text = normalize_text(r.get("text"))
        label = (r.get("label") or "").strip().capitalize()
        if not text:
            dropped.append({"row": i, "text": r.get("text", ""), "label": label, "reason": "missing text"})
        elif not label:
            dropped.append({"row": i, "text": text, "label": "", "reason": "missing label"})
        elif label not in COMMANDS:
            dropped.append({"row": i, "text": text, "label": label, "reason": "invalid label"})
        else:
            candidates.append({"row": i, "text": text, "label": label})

    labels_per_text = {}
    for c in candidates:
        labels_per_text.setdefault(c["text"], set()).add(c["label"])

    clean, seen = [], set()
    for c in candidates:
        if len(labels_per_text[c["text"]]) > 1:
            dropped.append({**c, "reason": "conflicting labels"})
        elif c["text"] in seen:
            dropped.append({**c, "reason": "duplicate"})
        else:
            seen.add(c["text"])
            clean.append(c)
    return clean, dropped


def split_80_20(clean):
    """Stratified 80/20 split, random_state=42. Adds 'split' = train/test to every row."""
    texts = [r["text"] for r in clean]
    labels = [r["label"] for r in clean]
    _, test_texts = train_test_split(texts, test_size=0.2, stratify=labels, random_state=42)
    test = set(test_texts)
    for r in clean:
        r["split"] = "test" if r["text"] in test else "train"
    return clean


# ------------------------------------------------------------------
# Metrics (same ones used for the AI models)
# ------------------------------------------------------------------
def command_metrics(pairs):
    """pairs = list of (expected, predicted). Returns None for an empty list
    (no division by zero). 'Unknown' counts as a miss.
    precision / recall / F1 are macro averages over the 4 commands."""
    n = len(pairs)
    if n == 0:
        return None
    correct = sum(1 for e, p in pairs if e == p)
    ps, rs, fs = [], [], []
    for c in COMMANDS:
        tp = sum(1 for e, p in pairs if e == c and p == c)
        pred = sum(1 for _, p in pairs if p == c)
        real = sum(1 for e, _ in pairs if e == c)
        pr = tp / pred if pred else 0.0
        rc = tp / real if real else 0.0
        ps.append(pr); rs.append(rc)
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"n": n, "correct": correct, "accuracy": correct / n,
            "precision": sum(ps) / 4, "recall": sum(rs) / 4, "f1": sum(fs) / 4}


def fmt(m, key):
    return "N/A" if m is None else f"{m[key]:.2f}"


# ------------------------------------------------------------------
# Main program
# ------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = load_csv(DATA_FILE)
    clean, dropped = preprocess(raw)
    clean = split_80_20(clean)
    for r in clean:
        r["predicted"], r["why"] = classify_command(r["text"])

    with open(os.path.join(OUT_DIR, "clean_phrases.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "split"])
        w.writerows([r["text"], r["label"], r["split"]] for r in clean)
    with open(os.path.join(OUT_DIR, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "expected", "predicted", "correct", "split", "rule_explanation"])
        w.writerows([r["text"], r["label"], r["predicted"], r["label"] == r["predicted"], r["split"], r["why"]]
                    for r in clean)

    line = "=" * 70
    print(line)
    print(" VOICECHECK - WEEK 2 NON-AI BASELINE (keyword rules on phrases.csv)")
    print(line)

    print("\n[1] Data check")
    print(f"    rows read          {len(raw):>4}")
    print(f"    rows kept          {len(clean):>4}")
    print(f"    rows dropped       {len(dropped):>4}")
    for reason, n in Counter(d["reason"] for d in dropped).items():
        print(f"      - {reason}: {n}")
    counts = Counter(r["label"] for r in clean)
    print("    phrases per command: " + ", ".join(f"{c} {counts[c]}" for c in COMMANDS))

    print("\n[2] 80/20 split (stratified, random_state=42)")
    for sp in ("train", "test"):
        c = Counter(r["label"] for r in clean if r["split"] == sp)
        print(f"    {sp:<5} {sum(c.values()):>4}   " + ", ".join(f"{k} {c[k]}" for k in COMMANDS))

    print("\n[3] Results")
    print(f"    {'set':<16}{'correct':>9}{'accuracy':>10}{'precision':>11}{'recall':>8}{'F1':>7}")
    for name, rows in (("all phrases", clean), ("20% test split", [r for r in clean if r["split"] == "test"])):
        m = command_metrics([(r["label"], r["predicted"]) for r in rows])
        corr = "-" if m is None else f"{m['correct']}/{m['n']}"
        print(f"    {name:<16}{corr:>9}{fmt(m,'accuracy'):>10}{fmt(m,'precision'):>11}"
              f"{fmt(m,'recall'):>8}{fmt(m,'f1'):>7}")

    print("\n[4] Confusion table, all phrases (rows = correct command, columns = baseline answer)")
    cols = list(COMMANDS) + ["Unknown"]
    print("    " + " " * 10 + "".join(f"{c:>10}" for c in cols))
    for e in COMMANDS:
        print(f"    {e:<10}" + "".join(
            f"{sum(1 for r in clean if r['label'] == e and r['predicted'] == p):>10}" for p in cols))

    print("\n[5] Wrong answers")
    for r in clean:
        if r["label"] != r["predicted"]:
            print(f"    [{r['split']:<5}] '{r['text']}'  expected {r['label']}, got {r['predicted']} ({r['why']})")
    print("\nSaved output/clean_phrases.csv and output/predictions.csv")
    print(line)


if __name__ == "__main__":
    main()