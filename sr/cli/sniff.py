# sr/cli/sniff.py
import argparse
import logging
from sr.core.sniff_orchestrator import infer_protocol, sniff
from sr.io.runs import Runs

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", type=str)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    runs = Runs(args.out)
    logging.getLogger("sniff").info(f"[cli] run dir: {runs.root}")

    draft = infer_protocol(args.prompt)
    if getattr(draft, "needs_reprompt", False):
        print("Protocol requires reprompt:", draft.reprompt_reason)
        return

    proto = draft
    proto, seeds = sniff(proto, runs)
    runs.save_json("sniff_protocol.json", proto.model_dump())
    runs.save_json("sniff_seeds.json", [s.model_dump() for s in seeds])

    # Point to debug artifacts, if present
    print(f"SNIFF done. Seeds: {len(seeds)} | out={runs.root}")
    print(f"  Look for: retrieval_debug.json, sniff_candidates.tsv, pass_a_debug.tsv (if generated)")

if __name__ == "__main__":
    main()
