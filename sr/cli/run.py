# sr/cli/run.py
import argparse
from sr.core.engine import run_end_to_end

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", type=str)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    ledger = run_end_to_end(args.prompt, args.out)
    print(f"End-to-end done. Screened: {len(ledger)}")

if __name__ == "__main__":
    main()
