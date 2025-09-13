# sr/cli/commit.py
import argparse
from sr.core.commit_orchestrator import commit
from sr.core.sniff_orchestrator import infer_protocol
from sr.io.runs import Runs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", type=str)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    runs = Runs(args.out)
    draft = infer_protocol(args.prompt)
    if getattr(draft, "needs_reprompt", False):
        raise SystemExit(f"Protocol needs reprompt: {draft.reprompt_reason}")
    ledger, diary = commit(draft, args.prompt, out_dir=str(runs.root))
    runs.save_json("protocol.json", draft.model_dump())
    runs.save_json("search_diary.json", diary.snapshot().model_dump())
    print(f"COMMIT done. Screened: {len(ledger)} | out={runs.root}")

if __name__ == "__main__":
    main()
