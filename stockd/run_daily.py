"""
Orchestrator daily:
- rulează BVB agent (RO)
- rulează Yahoo agent (EU + US)
Este punctul unic de intrare pentru workflow-ul Daily Prices Sync.
"""

from agents import bvb_agent, yahoo_agent


def main():
    print("[RUN] Starting daily sync (RO + EU + US)")

    # 1. România – BVB
    try:
        print("[RUN] Step 1/2: BVB agent (RO)…")
        bvb_agent.main()
    except Exception as e:
        print("[RUN][ERROR] BVB agent failed:", e)

    # 2. Global – Yahoo (EU + US)
    try:
        print("[RUN] Step 2/2: Yahoo agent (EU + US)…")
        yahoo_agent.main()
    except Exception as e:
        print("[RUN][ERROR] Yahoo agent failed:", e)

    print("[RUN] Daily sync finished.")


if __name__ == "__main__":
    main()
