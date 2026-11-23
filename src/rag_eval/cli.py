"""``rag-eval`` command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from rag_eval.config import Settings
from rag_eval.evals.compare import compare_results, render_comparison
from rag_eval.evals.golden import load_golden
from rag_eval.evals.report import render_report
from rag_eval.evals.runner import run_eval
from rag_eval.generation.claude import LLMError
from rag_eval.http import ProviderError
from rag_eval.pipeline import (
    ConfigError,
    build_embedder,
    build_generator,
    build_judge,
    build_reranker,
    build_retriever,
    build_store,
    ingest_corpus,
)
from rag_eval.retrieval.hybrid import RetrievalMode
from rag_eval.store.memory import MemoryStore

app = typer.Typer(
    help="RAG pipeline + evaluation harness with a CI regression gate.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

EmbedderOpt = Annotated[
    str | None, typer.Option(help="hashing | openai | voyage (env: RAG_EMBEDDER)")
]
StoreOpt = Annotated[str | None, typer.Option(help="memory | pgvector (env: RAG_STORE)")]
ChunkerOpt = Annotated[str | None, typer.Option(help="recursive | semantic (env: RAG_CHUNKER)")]
ChunkSizeOpt = Annotated[int | None, typer.Option(help="Max characters per chunk.")]
OverlapOpt = Annotated[int | None, typer.Option(help="Characters shared by adjacent chunks.")]
LLMOpt = Annotated[str | None, typer.Option(help="fake | anthropic (env: RAG_LLM)")]
ModeOpt = Annotated[str, typer.Option(help="hybrid | vector | keyword")]
RerankOpt = Annotated[str, typer.Option(help="none | voyage")]


def _settings(**overrides: Any) -> Settings:
    base = Settings()
    update = {k: v for k, v in overrides.items() if v is not None}
    try:
        return Settings.model_validate({**base.model_dump(), **update})
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _mode(value: str) -> RetrievalMode:
    if value not in ("hybrid", "vector", "keyword"):
        raise typer.BadParameter(f"invalid mode {value!r}; expected hybrid, vector or keyword")
    return value  # type: ignore[return-value]  # narrowed by the membership check above


def _fail(message: str) -> typer.Exit:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=2)


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def ingest(
    corpus: Annotated[Path, typer.Option(help="Directory of .md/.txt files.")] = Path("corpus"),
    store: StoreOpt = None,
    embedder: EmbedderOpt = None,
    chunker: ChunkerOpt = None,
    chunk_size: ChunkSizeOpt = None,
    chunk_overlap: OverlapOpt = None,
    reset: Annotated[bool, typer.Option(help="Clear the store before ingesting.")] = True,
) -> None:
    """Chunk, embed and index a corpus."""
    settings = _settings(
        store=store,
        embedder=embedder,
        chunker=chunker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    try:
        target = build_store(settings, fresh=reset)
        if reset:
            target.clear()
        stats = ingest_corpus(settings, corpus, target)
    except (ConfigError, ProviderError, FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    if isinstance(target, MemoryStore):
        target.save(Path(settings.index_path), embedder=build_embedder(settings).name)
        where = settings.index_path
    else:
        where = f"pgvector table {settings.pg_table}"
    typer.echo(f"Indexed {stats.chunks} chunks from {stats.documents} documents -> {where}")


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Natural-language question.")],
    k: Annotated[int | None, typer.Option("--k", help="Chunks to retrieve.")] = None,
    mode: ModeOpt = "hybrid",
    reranker: RerankOpt = "none",
    store: StoreOpt = None,
    embedder: EmbedderOpt = None,
    llm: LLMOpt = None,
    generate: Annotated[bool, typer.Option(help="Generate a cited answer.")] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Retrieve (and optionally answer) a single question against an ingested index."""
    settings = _settings(store=store, embedder=embedder, llm=llm, top_k=k)
    try:
        target = build_store(settings)
        if target.count() == 0:
            raise ConfigError("the index is empty; run `rag-eval ingest` first")
        retriever = build_retriever(
            settings,
            build_embedder(settings),
            target,
            mode=_mode(mode),
            reranker=build_reranker(settings, reranker),
        )
        hits = retriever.retrieve(question, settings.top_k)
        answer = build_generator(settings).generate(question, hits) if generate else None
    except (ConfigError, ProviderError, LLMError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    if as_json:
        payload: dict[str, Any] = {
            "question": question,
            "retrieved": [{"id": h.chunk.id, "score": h.score, "text": h.chunk.text} for h in hits],
        }
        if answer is not None:
            payload["answer"] = {
                "text": answer.text,
                "citations": list(answer.citations),
                "abstained": answer.abstained,
                "model": answer.model,
            }
        typer.echo(json.dumps(payload, indent=2))
        return

    for rank, hit in enumerate(hits, start=1):
        preview = " ".join(hit.chunk.text.split())[:110]
        typer.echo(f"{rank:>2}. [{hit.chunk.id}] {hit.score:.4f}  {preview}")
    if answer is not None:
        typer.echo(f"\n{answer.text}")
        typer.echo(f"citations: {', '.join(answer.citations) or '(none)'}  ·  {answer.model}")


@app.command()
def run(
    golden: Annotated[Path, typer.Option(help="Golden-set YAML.")] = Path("evals/golden.yaml"),
    corpus: Annotated[Path, typer.Option(help="Corpus to ingest.")] = Path("corpus"),
    out: Annotated[Path, typer.Option(help="Result JSON path.")] = Path("results/current.json"),
    report: Annotated[Path, typer.Option(help="Markdown report path.")] = Path("results/report.md"),
    write_report: Annotated[
        bool, typer.Option("--report-md/--no-report-md", help="Write the markdown report.")
    ] = True,
    k: Annotated[int | None, typer.Option("--k", help="Chunks passed to the generator.")] = None,
    mode: ModeOpt = "hybrid",
    reranker: RerankOpt = "none",
    store: StoreOpt = None,
    embedder: EmbedderOpt = None,
    chunker: ChunkerOpt = None,
    chunk_size: ChunkSizeOpt = None,
    chunk_overlap: OverlapOpt = None,
    llm: LLMOpt = None,
    generate: Annotated[bool, typer.Option(help="Run generation metrics.")] = True,
    judge: Annotated[bool, typer.Option(help="Run the LLM judge.")] = True,
    sanity_cases: Annotated[int, typer.Option(help="Judge sanity-check cases (0 = off).")] = 10,
) -> None:
    """Ingest the corpus into a fresh index, run the golden set, write JSON + markdown."""
    settings = _settings(
        store=store,
        embedder=embedder,
        chunker=chunker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        llm=llm,
        top_k=k,
    )
    try:
        golden_set = load_golden(golden)
        target = build_store(settings, fresh=True)
        target.clear()
        stats = ingest_corpus(settings, corpus, target)
        emb = build_embedder(settings)
        retriever = build_retriever(
            settings, emb, target, mode=_mode(mode), reranker=build_reranker(settings, reranker)
        )
        generator = build_generator(settings) if generate else None
        judge_impl = build_judge(settings) if (generate and judge) else None
    except (ConfigError, ProviderError, FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    config = {
        "embedder": emb.name,
        "chunker": settings.chunker,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "store": settings.store,
        "mode": mode,
        "reranker": reranker,
        "top_k": settings.top_k,
        "generator": generator.name if generator else None,
        "judge": judge_impl.name if judge_impl else None,
        "chunks": stats.chunks,
    }

    def progress(record: dict[str, Any]) -> None:
        status = "ERR" if "error" in record else "ok "
        typer.echo(f"  {status} {record['id']}", err=True)

    try:
        result = run_eval(
            golden_set,
            retriever,
            generator=generator,
            judge=judge_impl,
            top_k=settings.top_k,
            sanity_cases=sanity_cases,
            config=config,
            on_item=progress,
        )
    except ProviderError as exc:
        raise _fail(str(exc)) from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if write_report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(result), encoding="utf-8")

    width = max(len(n) for n in result["metrics"]) if result["metrics"] else 0
    for name, value in result["metrics"].items():
        typer.echo(f"{name:<{width}}  {value:.3f}")
    typer.echo(f"\nwrote {out}" + (f" and {report}" if write_report else ""))
    if result["run"]["n_errors"]:
        typer.secho(
            f"{result['run']['n_errors']} item(s) errored; see {out}", fg=typer.colors.YELLOW
        )


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(help="Baseline result JSON.")],
    current: Annotated[Path, typer.Argument(help="Current result JSON.")],
    fail_on_regression: Annotated[
        float | None,
        typer.Option(help="Exit 1 if any metric drops by more than this (absolute)."),
    ] = None,
    only: Annotated[
        list[str] | None, typer.Option(help="Restrict to metric prefixes, e.g. retrieval.")
    ] = None,
    summary: Annotated[
        Path | None, typer.Option(help="Append the table to this file (e.g. $GITHUB_STEP_SUMMARY).")
    ] = None,
) -> None:
    """Diff two eval results; with --fail-on-regression this is the CI gate."""
    try:
        base = json.loads(baseline.read_text(encoding="utf-8"))
        cur = json.loads(current.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail(str(exc)) from exc
    threshold = fail_on_regression if fail_on_regression is not None else 0.0
    cmp = compare_results(base, cur, threshold=threshold, prefixes=tuple(only or ()))
    table = render_comparison(cmp)
    typer.echo(table)
    if summary is not None:
        with summary.open("a", encoding="utf-8") as fh:
            fh.write(table + "\n")
    if fail_on_regression is not None and cmp.regressions:
        names = ", ".join(d.name for d in cmp.regressions)
        typer.secho(f"regression gate failed: {names}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
