import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from functools import lru_cache
import re
from typing import Any, cast

from cleantext.clean import clean
from fastapi import FastAPI, Request
from loguru import logger
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
from pydantic import BaseModel, Field
from setfit import SetFitModel
from transformers import (
    AutoTokenizer,
    pipeline,
)

from beacon_analysis_server.config import EXTENDED_STOP_WORDS, MODELS_DIR, RCA_TOP_N, SENTIMENT_MAP

STOP_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, EXTENDED_STOP_WORDS)) + r")\b", re.IGNORECASE
)

RE_FLOOD = re.compile(r"(.)\1{2,}")  # matches 3 or more repeated characters
RE_ADVERSARIAL = re.compile(r"[\._]+")  # matches sequences of . or _ characters


class Report(BaseModel):
    """
    A user report containing a description of their issue.
    """

    description: str = Field(..., min_length=3, max_length=500)


class Sentiment(BaseModel):
    """
    The sentiment analysis result for a user report.
    """

    category: str
    confidence_score: float


class ReportList(BaseModel):
    """
    A list of user report descriptions for batch processing.
    """

    descriptions: list[str] = Field(..., min_length=1)


class RCAItem(BaseModel):
    """
    An individual root cause analysis finding.
    """

    root_cause: str
    confidence_score: float


class RCASummary(BaseModel):
    """
    The summary of root cause analysis across a batch of reports.
    """

    total_processed: int
    top_issue: str
    top_findings: list[RCAItem]


@lru_cache(maxsize=512)
def clean_report(
    text: str,
    re_flood=RE_FLOOD,
    re_adversarial=RE_ADVERSARIAL,
    stop_pattern=STOP_PATTERN,
    deep_clean=False,
) -> str:
    """
    Cleans the input text by removing stop words, normalizing characters, and reducing noise.
    """

    logger.debug(f"[clean] Original text: {text}")
    if deep_clean:
        text = clean(
            text,
            fix_unicode=True,
            to_ascii=True,
            lower=True,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=True,
            no_punct=False,
            lang="en",
        )
        text = stop_pattern.sub("", text)

    # soooooo / !!!!! / hahahahaha
    text = re_flood.sub(r"\1", text)
    logger.debug(f"[clean] Scrubbed: {text}")

    # sh_t / wala.naman kuryente...
    text = re_adversarial.sub("", text)
    logger.debug(f"[clean] New scrubbed: {text}")

    return text


async def batch_processor(
    queue: asyncio.Queue,
    pipeline: Any,
    batch_size: int = 12,
    timeout: float = 1.0,
    sentiment_map: dict[str, tuple[str, int]] = SENTIMENT_MAP,
):
    """
    Groups individual reports into batches for high-throughput inference.
    """

    while True:
        batch: list[tuple[str, asyncio.Future[Sentiment]]] = list()
        item = await queue.get()
        batch.append(item)

        # Gather more items up to batch_size or until timeout
        end_time = asyncio.get_event_loop().time() + timeout
        while len(batch) < batch_size:
            wait_time = end_time - asyncio.get_event_loop().time()
            if wait_time <= 0:
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=wait_time)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        raw_texts = [b[0] for b in batch]
        clean_texts = [clean_report(text) for text in raw_texts]

        results = pipeline(clean_texts)

        logger.debug(f"Raw Output: {results}")
        for i, (_, future) in enumerate(batch):
            if i >= len(results):
                continue

            raw = results[i]
            label = raw["label"]
            score = raw["score"]

            cat_str, _ = sentiment_map.get(label, ("Neutral", 0))

            if not future.done():
                future.set_result(
                    Sentiment(
                        category=cat_str,
                        confidence_score=score,
                    )
                )


def make_pipeline(
    id: str,
    model_name: str,
    session_options: ort.SessionOptions,
    batch_size: int,
    pipeline_name: Any,
):
    """
    Creates a text classification pipeline using the specified model and tokenizer.
    """

    model = ORTModelForSequenceClassification.from_pretrained(
        id, file_name=model_name, local_files_only=True, session_options=session_options
    )
    tokenizer = AutoTokenizer.from_pretrained(id, file_name=model_name, fix_mistral_regex=True)
    return pipeline(
        pipeline_name,
        model=cast(Any, model),
        tokenizer=tokenizer,
        device="cpu",
        batch_size=batch_size,
    )


@asynccontextmanager
async def lifespan(app: FastAPI, models_dir=MODELS_DIR):
    """
    Manages the lifespan of the FastAPI application, including setup and teardown of resources.
    """

    queue: asyncio.Queue[tuple[str, asyncio.Future[Sentiment]]] = asyncio.Queue()

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    roberta_classifier = make_pipeline(
        id=f"{models_dir}/roberta_sentiment_custom/",
        model_name="model.onnx",
        session_options=sess_options,
        batch_size=24,
        pipeline_name="text-classification",
    )

    rca_model = SetFitModel.from_pretrained(f"{models_dir}/setfit_v2", local_files_only=True)

    app.state.report_queue = queue
    app.state.roberta_classifier = roberta_classifier
    app.state.rca_model = rca_model

    processor_task = asyncio.create_task(batch_processor(queue, roberta_classifier))
    yield
    processor_task.cancel()


app = FastAPI(
    title="Beacon Analysis Server",
    version="0.3.0",
    description="A FastAPI server serving Beacon analysis logic",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """
    Health check endpoint to verify the server is running.
    """

    return {"message": "Beacon Analysis Server is online"}


@app.post("/reports/sentiment", response_model=Sentiment)
async def run_analysis(report: Report, request: Request):
    """
    Analyzes a single description from a user report and returns the sentiment (NEGATIVE, NEUTRAL, POSITIVE) and confidence.
    """
    future: asyncio.Future[Sentiment] = asyncio.get_running_loop().create_future()
    report_queue: asyncio.Queue = request.app.state.report_queue
    await report_queue.put((report.description, future))

    result = await future
    logger.debug(f"Result for '{report.description}': {result}")

    return result


@app.post("/reports/batch/rca", response_model=RCASummary)
async def run_batch_rca(data: ReportList, request: Request):
    """
    Analyzes a collection of reports to identify the underlying
    failure or root cause across the batch.
    """
    model: SetFitModel = request.app.state.rca_model

    cleaned = [clean_report(r) for r in data.descriptions]
    predictions: list[str] = cast(list[str], model(cleaned))

    label_counts = Counter(predictions)
    sorted_labels = label_counts.most_common(RCA_TOP_N)

    top_findings = [
        RCAItem(
            root_cause=label,
            confidence_score=round(count / len(cleaned), 4),
        )
        for label, count in sorted_labels
        if count > 0
    ]

    top_issue = top_findings[0].root_cause if top_findings else "General Grid Issue"

    logger.debug(f"RCA Results: {label_counts.most_common(5)}")
    return RCASummary(
        total_processed=len(cleaned),
        top_issue=top_issue,
        top_findings=top_findings,
    )
