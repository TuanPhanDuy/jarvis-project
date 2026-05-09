"""CLI entrypoint for JARVIS eval runner.

Usage:
    jarvis-eval                          # run baseline suite
    jarvis-eval --suite path/to/suite.json
    jarvis-eval --tags ml basics         # filter by tags
    jarvis-eval --judge                  # enable Claude-as-judge scoring
    jarvis-eval --output results.json    # save detailed results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="JARVIS Eval Runner")
    parser.add_argument("--suite", metavar="FILE", help="Path to eval suite JSON (default: built-in baseline)")
    parser.add_argument("--tags", nargs="*", metavar="TAG", help="Filter cases by tag")
    parser.add_argument("--judge", action="store_true", help="Enable Claude-as-judge scoring")
    parser.add_argument("--output", metavar="FILE", help="Save full results to JSON file")
    parser.add_argument("--no-persist", action="store_true", help="Skip auto-saving to eval_history.jsonl")
    parser.add_argument("--analyze-feedback", action="store_true", help="Run self-improvement feedback analysis")
    args = parser.parse_args()

    try:
        from jarvis.config import get_settings
        settings = get_settings()
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.analyze_feedback:
        from jarvis.evals.feedback_analyzer import run_analysis
        db_path = settings.reports_dir / "jarvis.db"
        print("Running feedback analysis…")
        result = run_analysis(db_path, settings.reports_dir, model=settings.model)
        print(f"Done: {result}")
        sys.exit(0)

    from jarvis.evals.suite import BASELINE_SUITE, load_suite
    from jarvis.evals.runner import run_suite, summarize, persist_results

    cases = load_suite(Path(args.suite)) if args.suite else BASELINE_SUITE
    print(f"Running {len(cases)} eval case(s)...\n")

    results = run_suite(cases, settings, use_judge=args.judge, tags_filter=args.tags)

    # Print per-case results
    for r in results:
        status = "PASS" if r.overall_pass else "FAIL"
        judge_str = f"  judge={r.judge_score}/5" if r.judge_score else ""
        print(f"  [{status}] {r.case_id}  ({r.latency_s}s  ${r.cost_usd:.5f}){judge_str}")
        if r.failed_contains:
            print(f"    missing: {r.failed_contains}")
        if r.failed_forbidden:
            print(f"    forbidden found: {r.failed_forbidden}")
        if r.error:
            print(f"    error: {r.error}")
        if r.judge_reasoning:
            print(f"    reasoning: {r.judge_reasoning}")

    summary = summarize(results)
    print(f"\nSummary: {summary['passed']}/{summary['total']} passed "
          f"({summary['pass_rate']*100:.1f}%)  "
          f"avg {summary['avg_latency_s']}s  "
          f"total ${summary['total_cost_usd']:.5f}")
    if summary["avg_judge_score"]:
        print(f"  Avg judge score: {summary['avg_judge_score']}/5")

    if not args.no_persist:
        persist_results(results, summary, settings.reports_dir)

    if args.output:
        import dataclasses
        out = {
            "summary": summary,
            "results": [dataclasses.asdict(r) for r in results],
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output}")

    sys.exit(0 if summary["passed"] == summary["total"] else 1)


if __name__ == "__main__":
    main()
