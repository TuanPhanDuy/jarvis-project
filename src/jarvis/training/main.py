"""jarvis-train — CLI for the JARVIS AI training pipeline.

Subcommands:
  crawl      Fetch AI research from ArXiv, HF blog, Anthropic, Papers With Code
  generate   Synthesise instruction Q&A pairs from ingested docs (via Claude)
  finetune   LoRA fine-tune the local Ollama model via mlx-lm (Apple Silicon)
  register   Build an Ollama Modelfile and register the fine-tuned model
  eval       Run the JARVIS eval suite against a specified model
  pipeline   Run all 5 steps in sequence

Usage examples:
  jarvis-train crawl --topics "RLHF,transformers" --max 20
  jarvis-train generate --target-pairs 500
  jarvis-train finetune --epochs 3 --lora-rank 16
  jarvis-train register --name jarvis-ft
  jarvis-train eval --model jarvis-ft --judge
  jarvis-train pipeline --topics "RLHF" --name jarvis-ft
"""
from __future__ import annotations

import argparse
import sys


def _cmd_crawl(args: argparse.Namespace) -> None:
    from pathlib import Path
    from jarvis.config import get_settings
    from jarvis.training.crawler import ResearchCrawler

    settings = get_settings()
    crawler = ResearchCrawler(settings.reports_dir)
    sources = [s.strip() for s in args.sources.split(",")]
    topics = [t.strip() for t in args.topics.split(",")]

    total = 0
    for topic in topics:
        if "arxiv" in sources:
            docs = crawler.crawl_arxiv(topic, max_results=args.max)
            names = crawler.ingest_all(docs)
            total += len([n for n in names if not n.startswith("ERROR")])
            print(f"[arxiv] {topic}: ingested {len(docs)} docs")

        if "hf" in sources:
            docs = crawler.crawl_hf_blog(max_posts=args.max)
            names = crawler.ingest_all(docs)
            total += len([n for n in names if not n.startswith("ERROR")])
            print(f"[hf_blog] ingested {len(docs)} docs")

        if "anthropic" in sources:
            docs = crawler.crawl_anthropic(max_posts=args.max)
            names = crawler.ingest_all(docs)
            total += len([n for n in names if not n.startswith("ERROR")])
            print(f"[anthropic] ingested {len(docs)} docs")

        if "pwc" in sources:
            docs = crawler.crawl_papers_with_code(topic, max_results=args.max)
            names = crawler.ingest_all(docs)
            total += len([n for n in names if not n.startswith("ERROR")])
            print(f"[pwc] {topic}: ingested {len(docs)} docs")

    print(f"\nTotal ingested: {total} documents")


def _cmd_generate(args: argparse.Namespace) -> None:
    from pathlib import Path
    from jarvis.config import get_settings
    from jarvis.training.data_generator import TrainingDataGenerator
    from jarvis.training.dataset_manager import DatasetManager

    settings = get_settings()
    data_dir = Path(settings.training_data_dir)
    dataset_path = data_dir / "dataset.jsonl"

    generator = TrainingDataGenerator(
        reports_dir=settings.reports_dir,
        api_key=settings.anthropic_api_key,
    )
    n = generator.run(
        out_path=dataset_path,
        target_pairs=args.target_pairs,
        pairs_per_chunk=args.pairs_per_chunk,
    )

    dm = DatasetManager()
    removed = dm.deduplicate(dataset_path)
    stats = dm.stats(dataset_path)
    print(f"Generated {n} pairs, removed {removed} duplicates.")
    print(f"Dataset: {stats['count']} pairs, avg {stats.get('avg_chars', 0)} chars each")
    print(f"Path: {dataset_path}")


def _cmd_finetune(args: argparse.Namespace) -> None:
    from pathlib import Path
    from jarvis.config import get_settings
    from jarvis.training.dataset_manager import DatasetManager
    from jarvis.training.finetune import Finetuner

    settings = get_settings()
    data_dir = Path(settings.training_data_dir)
    dataset_path = data_dir / "dataset.jsonl"

    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}. Run `jarvis-train generate` first.")
        sys.exit(1)

    dm = DatasetManager()
    train_path, val_path = dm.split(dataset_path)
    print(f"Dataset split: train={train_path}, val={val_path}")

    adapter_dir = data_dir / "adapters"
    finetuner = Finetuner(
        base_model=settings.training_base_model_mlx,
        adapter_dir=adapter_dir,
    )
    finetuner.train(
        data_dir=data_dir,
        epochs=args.epochs,
        lora_rank=args.lora_rank,
    )
    print(f"Fine-tuning complete. Adapters at: {adapter_dir}")


def _cmd_register(args: argparse.Namespace) -> None:
    from pathlib import Path
    from jarvis.config import get_settings
    from jarvis.training.finetune import Finetuner
    from jarvis.training.modelfile import register_model

    settings = get_settings()
    data_dir = Path(settings.training_data_dir)
    gguf_path = data_dir / f"{args.name}.gguf"

    if not gguf_path.exists():
        adapter_dir = data_dir / "adapters"
        if not adapter_dir.exists():
            print(f"ERROR: no adapters found at {adapter_dir}. Run `jarvis-train finetune` first.")
            sys.exit(1)
        print(f"Exporting GGUF to {gguf_path}...")
        finetuner = Finetuner(
            base_model=settings.training_base_model_mlx,
            adapter_dir=adapter_dir,
        )
        finetuner.export_gguf(gguf_path)

    ok = register_model(gguf_path, args.name)
    if ok:
        print(f"Model '{args.name}' registered. Use with: JARVIS_MODEL={args.name} uv run jarvis")
    else:
        print(f"ERROR: registration failed. Check ollama is running.")
        sys.exit(1)


def _cmd_eval(args: argparse.Namespace) -> None:
    import os
    from jarvis.config import get_settings
    from jarvis.evals.runner import run_suite, summarize
    from jarvis.evals.suite import BASELINE_SUITE

    settings = get_settings()
    model = args.model or settings.model
    os.environ["JARVIS_MODEL"] = model

    print(f"Running eval suite with model: {model}")
    results = run_suite(BASELINE_SUITE, judge=args.judge)
    summary = summarize(results)
    print(f"\nPass rate: {summary['pass_rate']:.1%}")
    print(f"Avg latency: {summary['avg_latency_s']:.2f}s")
    if summary.get("avg_judge_score"):
        print(f"Avg judge score: {summary['avg_judge_score']:.2f}/5")


def _cmd_pipeline(args: argparse.Namespace) -> None:
    print("=== Step 1/5: Crawl ===")
    _cmd_crawl(args)
    print("\n=== Step 2/5: Generate ===")
    _cmd_generate(args)
    print("\n=== Step 3/5: Fine-tune ===")
    _cmd_finetune(args)
    print("\n=== Step 4/5: Register ===")
    _cmd_register(args)
    print("\n=== Step 5/5: Eval ===")
    args.model = args.name
    _cmd_eval(args)
    print("\n=== Pipeline complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jarvis-train",
        description="JARVIS AI training pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── crawl ──────────────────────────────────────────────────────────────
    p_crawl = sub.add_parser("crawl", help="Crawl AI research from the internet")
    p_crawl.add_argument("--topics", default="RLHF,transformers,constitutional AI",
                         help="Comma-separated research topics")
    p_crawl.add_argument("--sources", default="arxiv,hf,anthropic,pwc",
                         help="Comma-separated sources: arxiv,hf,anthropic,pwc")
    p_crawl.add_argument("--max", type=int, default=10,
                         help="Max documents per source per topic")
    p_crawl.set_defaults(func=_cmd_crawl)

    # ── generate ───────────────────────────────────────────────────────────
    p_gen = sub.add_parser("generate", help="Generate instruction Q&A pairs via Claude")
    p_gen.add_argument("--target-pairs", type=int, default=500,
                       help="Target number of Q&A pairs")
    p_gen.add_argument("--pairs-per-chunk", type=int, default=3,
                       help="Pairs to generate per document chunk")
    p_gen.set_defaults(func=_cmd_generate)

    # ── finetune ───────────────────────────────────────────────────────────
    p_ft = sub.add_parser("finetune", help="LoRA fine-tune via mlx-lm (Apple Silicon)")
    p_ft.add_argument("--epochs", type=int, default=3)
    p_ft.add_argument("--lora-rank", type=int, default=16)
    p_ft.set_defaults(func=_cmd_finetune)

    # ── register ───────────────────────────────────────────────────────────
    p_reg = sub.add_parser("register", help="Register fine-tuned model with Ollama")
    p_reg.add_argument("--name", default="jarvis-ft",
                       help="Ollama model name to register")
    p_reg.set_defaults(func=_cmd_register)

    # ── eval ───────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evaluate a model with the JARVIS eval suite")
    p_eval.add_argument("--model", default="",
                        help="Ollama model name to evaluate (default: JARVIS_MODEL)")
    p_eval.add_argument("--judge", action="store_true",
                        help="Enable Ollama-as-judge scoring")
    p_eval.set_defaults(func=_cmd_eval)

    # ── pipeline ───────────────────────────────────────────────────────────
    p_pipe = sub.add_parser("pipeline", help="Run the full pipeline: crawl→generate→finetune→register→eval")
    p_pipe.add_argument("--topics", default="RLHF,transformers,constitutional AI")
    p_pipe.add_argument("--sources", default="arxiv,hf,anthropic,pwc")
    p_pipe.add_argument("--max", type=int, default=10)
    p_pipe.add_argument("--target-pairs", type=int, default=500)
    p_pipe.add_argument("--pairs-per-chunk", type=int, default=3)
    p_pipe.add_argument("--epochs", type=int, default=3)
    p_pipe.add_argument("--lora-rank", type=int, default=16)
    p_pipe.add_argument("--name", default="jarvis-ft")
    p_pipe.add_argument("--judge", action="store_true")
    p_pipe.set_defaults(func=_cmd_pipeline)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
