"""A small look at follow-up questions with the real language model (the free providers in .env).

Each case is a short conversation on one corpus paper (corpus/papers.json; the PDFs in --papers-dir
as for eval_retrieval.py). The script asks the questions in turn through the app, as a visitor would,
and prints for each follow-up: how the agent understood it, whether that is right (it contains the
words it should, or, for a question that is already complete, it is unchanged) and the checker's
line. A handful of cases, not a benchmark.

    python eval_followup.py --papers-dir papers
"""
import argparse
import os
import sys
import warnings

SAME = "unchanged"
CASES = [  # (paper key, [(question, words its rewrite should contain, or SAME), ...])
    ("wootters1998", [("How is the concurrence of a two-qubit state computed?", None),
                      ("and how is the entanglement of formation obtained from it?", ["concurrence"]),
                      ("What is the entanglement of formation of a Bell state?", SAME)]),
    ("horodecki1996sep", [("What is the partial transposition criterion?", None),
                          ("for which dimensions is it also sufficient?", ["partial transpos"]),
                          ("and why does it fail in higher dimensions?", ["partial transpos"])]),
    ("ekert1991", [("How does the protocol detect an eavesdropper?", None),
                   ("which inequality is used for that?", ["eavesdropp"]),
                   ("what value does quantum mechanics predict for it?", ["inequality"])]),
    ("james2001", [("How many measurements are needed for the tomography of two qubits?", None),
                   ("and how is the density matrix reconstructed from them?", ["density matrix"]),
                   ("What is the maximum likelihood technique?", SAME)]),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", default="papers")
    ap.add_argument("--source", default="best", choices=["arxiv", "journal", "best"])
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv()
    warnings.filterwarnings("ignore", message="Local mode is not recommended")
    import agent
    import app
    import session
    from eval_retrieval import pdf_for
    right = total = 0
    for key, conversation in CASES:
        path = pdf_for(key, a.source, a.papers_dir)
        if not path:
            print(f"\n{key}: skipped, no PDF in {a.papers_dir}")
            continue
        sid = "followup_" + os.urandom(4).hex()
        try:
            with session.use(sid) as s:
                paper, msg = app.add_paper(path, s)
                if paper is None:
                    print(f"\n{key}: {msg}")
                    continue
                print(f"\n{key} ({paper.short_name})")
                for n, (q, want) in enumerate(conversation, start=1):
                    before = len(s.turns)
                    md, info, _, _ = app.answer_question(s, q, "all papers", use_background=False)
                    if len(s.turns) == before:
                        print(f"  {n}. {q}\n     not answered: {md[:200]}")
                        break
                    t = s.turns[-1]
                    print(f"  {n}. {q}")
                    if want is not None:
                        ok = agent._same(t["standalone"], q) if want == SAME else \
                            all(w.lower() in t["standalone"].lower() for w in want)
                        right, total = right + ok, total + 1
                        print(f"     understood as: {t['standalone']}\n"
                              f"     {'unchanged' if want == SAME else 'contains ' + str(want)}: "
                              f"{'yes' if ok else 'NO'}")
                    print(f"     {info.strip('*')}\n     answer: {md[:300]}")
        finally:
            session.drop(sid)
    print(f"\nfollow-ups understood as expected: {right} of {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
