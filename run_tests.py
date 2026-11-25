"""
VoiceCheck - Week 2 tests.
Each test has an EXPECTED value worked out by hand and an ACTUAL value
produced by baseline.py. Results are saved to Week2_Test_Log.csv.

Run:  python run_tests.py
"""
import csv
import os

from baseline import classify_command, command_metrics, preprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def cmd(phrase):
    return classify_command(phrase)[0]


def prep(rows):
    """Runs preprocessing on a tiny hand-made table and describes the result."""
    clean, dropped = preprocess([{"text": t, "label": l} for t, l in rows])
    kept = ", ".join(f"'{r['text']}'" for r in clean) or "none"
    why = ", ".join(sorted({d["reason"] for d in dropped})) or "none"
    return f"kept {len(clean)} ({kept}); dropped {len(dropped)} ({why})"


def acc(pairs):
    m = command_metrics(pairs)
    return "N/A" if m is None else f"{m['accuracy']:.2f}"


# (id, category, description, input, expected, function producing actual)
TESTS = [
    ("T01", "Normal", "Clear Register phrase", "'register my attendance'", "Register",
     lambda: cmd("register my attendance")),
    ("T02", "Normal", "Clear Repeat phrase (2 keyword hits)", "'say that again'", "Repeat",
     lambda: cmd("say that again")),
    ("T03", "Normal", "Confusion phrase (AI model got this wrong in Week 1)", "'i am confused'", "Help",
     lambda: cmd("i am confused")),
    ("T04", "Normal", "Accuracy on 4 answers, 3 correct",
     "(Register,Register) (Help,Help) (Cancel,Cancel) (Repeat,Unknown)", "0.75",
     lambda: acc([("Register", "Register"), ("Help", "Help"), ("Cancel", "Cancel"), ("Repeat", "Unknown")])),
    ("T05", "Missing/zero", "Empty phrase", "''", "Unknown", lambda: cmd("")),
    ("T06", "Missing/zero", "Row with no text is removed", "('', Help), ('help', Help)",
     "kept 1 ('help'); dropped 1 (missing text)", lambda: prep([("", "Help"), ("help", "Help")])),
    ("T07", "Missing/zero", "Row with no label is removed", "('stop', ''), ('cancel', Cancel)",
     "kept 1 ('cancel'); dropped 1 (missing label)", lambda: prep([("stop", ""), ("cancel", "Cancel")])),
    ("T08", "Missing/zero", "Accuracy of an empty list (no division by zero)", "[]", "N/A",
     lambda: acc([])),
    ("T09", "Missing/zero", "CSV with only the header row", "no rows",
     "kept 0 (none); dropped 0 (none)", lambda: prep([])),
    ("T10", "Duplicate", "Same phrase twice with the same label", "('help', Help) x2",
     "kept 1 ('help'); dropped 1 (duplicate)", lambda: prep([("help", "Help"), ("help", "Help")])),
    ("T11", "Duplicate", "Duplicate hidden by capitals and spaces", "('Check In', Register), (' check  in ', Register)",
     "kept 1 ('check in'); dropped 1 (duplicate)",
     lambda: prep([("Check In", "Register"), (" check  in ", "Register")])),
    ("T12", "Duplicate", "Same phrase with two different labels", "('stop', Cancel), ('stop', Help)",
     "kept 0 (none); dropped 2 (conflicting labels)", lambda: prep([("stop", "Cancel"), ("stop", "Help")])),
    ("T13", "Edge", "Negation with capitals and punctuation", "'DON'T Register!!'", "Cancel",
     lambda: cmd("DON'T Register!!")),
    ("T14", "Edge", "Tie: one Register word and one Help word", "'how do i register'", "Unknown",
     lambda: cmd("how do i register")),
    ("T15", "Edge", "Tie between two commands", "'repeat and cancel'", "Unknown",
     lambda: cmd("repeat and cancel")),
    ("T16", "Edge", "'start' means Register but 'start over' means Cancel", "'start over'", "Cancel",
     lambda: cmd("start over")),
    ("T17", "Edge", "Misspelled label in the CSV", "('register', 'Registr')",
     "kept 0 (none); dropped 1 (invalid label)", lambda: prep([("register", "Registr")])),
    ("T18", "Known limit", "Misheard 'hep' should mean Help", "'hep'", "Help", lambda: cmd("hep")),
    ("T19", "Known limit", "Misheard 'check me inn' should mean Register", "'check me inn'", "Register",
     lambda: cmd("check me inn")),
]


def main():
    rows, passed = [], 0
    print("-" * 72)
    for tid, cat, desc, inp, exp, fn in TESTS:
        try:
            act = fn()
        except Exception as e:                      # a crash counts as a failed test
            act = f"ERROR: {e}"
        res = "PASS" if act == exp else "FAIL"
        passed += res == "PASS"
        rows.append([tid, cat, desc, inp, exp, act, res])
        print(f"{tid}  {res}  [{cat}] {desc}")
        print(f"            expected: {exp}")
        print(f"            actual:   {act}")
    print("-" * 72)
    print(f"{passed}/{len(TESTS)} passed, {len(TESTS) - passed} failed")
    with open(os.path.join(HERE, "Week2_Test_Log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test_id", "category", "description", "input", "expected", "actual", "result"])
        w.writerows(rows)
    print("Saved Week2_Test_Log.csv")


if __name__ == "__main__":
    main()