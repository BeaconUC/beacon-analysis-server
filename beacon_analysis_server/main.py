import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
import re
from typing import Optional

from cleantext.clean import clean
from fastapi import FastAPI, Request
from keybert import KeyBERT
from loguru import logger
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
from spacy.language import Language
import torch
from transformers import TextClassificationPipeline, pipeline

from beacon_analysis_server.config import MODELS_DIR

torch.set_num_threads(1)

RE_FLOOD = re.compile(r"(.)\1{2,}")
RE_ADVERSARIAL = re.compile(r"[\._]+")
SENTIMENT_MAP = {
    "LABEL_0": ("Negative", -1),
    "LABEL_1": ("Positive", 1),
    "LABEL_2": ("Neutral", 0),
}
PHRASES = {
    "EMERGENCY": [
        "poste na tumba",
        "putol na kawad",
        "nagliliyab na transformer",
        "may sumasabog",
        "may nagliliyab",
        "live wire",
        "kawad sa kalsada",
    ],
    "TECHNICAL": [
        "pumutok na transformer",
        "walang kuryente",
        "brownout pa rin",
        "pumutok na kwan",
        "spark sa poste",
    ],
    "FOLLOWUP": ["kanina pa", "update naman", "follow up ko lang", "gaano katagal"],
}
TECHNICAL_CANDIDATES = [
    "transformer explosion",
    "pumutok na transformer",
    "fallen pole",
    "poste na tumba",
    "broken wire",
    "putol na kawad",
    "sparking wire",
    "maintenance",
    "overload",
    "brownout",
    "lightning strike",
    "kidlat",
    "short circuit",
]
TAGALOG_STOP_WORDS = [
    "sa",
    "ng",
    "ang",
    "mga",
    "na",
    "si",
    "ni",
    "ay",
    "ito",
    "sila",
    "kami",
    "kaming",
    "ko",
    "lang",
    "dito",
    "samin",
    "po",
    "opo",
    "namin",
    "inyo",
    "inyong",
    "ba",
    "kasi",
    "yung",
    "kayo",
    "mo",
    "muna",
    "naman",
    "tapat",
    "bahay",
    "kalsada",
    "kanto",
    "paligid",
    "baka",
    "sana",
    "paki",
    "ngayon",
    "kanina",
    "mula",
    "noon",
    "diyan",
    "dito",
    "doon",
    "ano",
    "kwan",
]
EXTENDED_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(TAGALOG_STOP_WORDS))


class Report(BaseModel):
    description: str = Field(..., min_length=3, max_length=500)


class AnalysisResult(BaseModel):
    sentiment_category_str: str
    sentiment_category_int: int
    sentiment_score: float
    urgency_flag: bool
    urgency_match: Optional[str]
    priority_level: str


class BatchRCAInput(BaseModel):
    descriptions: list[str] = Field(..., min_length=1)


class RCAIssue(BaseModel):
    root_cause: str
    confidence_score: float
    relevance: str


class RCASummary(BaseModel):
    total_processed: int
    primary_issue: str
    top_findings: list[RCAIssue]


def detect_fuzzy_urgency(text: str, threshold=80.0, phrases: dict[str, list[str]] = PHRASES):
    """
    Checks if any word in the text resembles the urgency targets.
    Returns True if a match is > threshold (0-100).
    """
    text = text.lower()
    for category, phrase_list in phrases.items():
        res = process.extractOne(
            text, phrase_list, scorer=fuzz.token_set_ratio, score_cutoff=threshold
        )
        if res:
            return category, res[0]
    return None, None


def calculate_priority(intent_cat: Optional[str], cat_int: int, score: float) -> str:
    """
    Determines administrative priority based on sentiment and technical intent.
    """
    if intent_cat == "EMERGENCY":
        return "CRITICAL"
    if intent_cat == "TECHNICAL" or (cat_int == -1 and score > 0.9):
        return "HIGH"
    if intent_cat == "FOLLOW_UP" or cat_int == -1:
        return "MEDIUM"
    if cat_int == 1:
        return "LOW (Feedback)"
    return "ROUTINE"


@lru_cache(maxsize=256)
def clean_report(text: str, re_flood=RE_FLOOD, re_adversarial=RE_ADVERSARIAL) -> str:
    logger.debug(f"[clean] Original text: {text}")

    scrubbed = clean(
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

    scrubbed = re_flood.sub(r"\1", scrubbed)
    logger.debug(f"[clean] Scrubbed: {scrubbed}")

    if "." in scrubbed or "_" in scrubbed:
        scrubbed = re_adversarial.sub("", scrubbed)
        logger.debug(f"[clean] New scrubbed: {scrubbed}")

    return scrubbed


async def batch_processor(
    queue: asyncio.Queue,
    pipeline: TextClassificationPipeline,
    batch_size: int = 16,
    timeout: float = 0.05,
    sentiment_map: dict[str, tuple[str, int]] = SENTIMENT_MAP,
):
    """Groups individual reports into batches for high-throughput inference."""
    with torch.inference_mode():
        while True:
            batch: list[tuple[str, asyncio.Future[AnalysisResult]]] = list()
            item = await queue.get()
            batch.append(item)

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

            raw_texts = map(lambda b: b[0], batch)
            clean_texts = list(map(clean_report, raw_texts))

            results = pipeline(clean_texts)

            logger.debug(f"Raw Output: {results}")
            for i, (original_text, future) in enumerate(batch):
                raw = results[i]
                label = raw["label"]
                score = raw["score"]

                cat_str, cat_int = sentiment_map.get(label, ("Neutral", 0))
                intent_cat, matched_phrase = detect_fuzzy_urgency(original_text)
                priority = calculate_priority(intent_cat, cat_int, score)

                if not future.done():
                    future.set_result(
                        AnalysisResult(
                            sentiment_category_str=cat_str,
                            sentiment_category_int=cat_int,
                            sentiment_score=score,
                            urgency_flag=intent_cat is not None,
                            urgency_match=matched_phrase,
                            priority_level=priority,
                        )
                    )


@asynccontextmanager
async def lifespan(app: FastAPI, models_dir=MODELS_DIR):
    model_path = f"{models_dir}/itanong_roberta"
    queue: asyncio.Queue[tuple[str, asyncio.Future[AnalysisResult]]] = asyncio.Queue()
    pipe = pipeline("text-classification", model=model_path, device="cpu")
    logger.debug(f"Model Label Mapping: {pipe.model.config.id2label}")

    kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import os

        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    app.state.report_queue = queue
    app.state.kw_model = kw_model
    app.state.nlp = nlp

    processor_task = asyncio.create_task(batch_processor(queue, pipe))
    yield
    processor_task.cancel()


app = FastAPI(
    title="Beacon Analysis Server",
    version="0.1.0",
    description="A FastAPI server serving Beacon analysis logic",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"message": "Beacon Analysis Server is online"}


@app.post("/reports/analyze", response_model=AnalysisResult)
async def run_analysis(report: Report, request: Request):
    future: asyncio.Future[AnalysisResult] = asyncio.get_running_loop().create_future()
    queue: asyncio.Queue = request.app.state.report_queue
    await queue.put((report.description, future))

    result = await future
    logger.debug(f"Result for '{report.description}': {result}")

    return result


@app.post("/reports/batch/rca", response_model=RCASummary)
async def run_batch_rca(data: BatchRCAInput, request: Request):
    """
    Analyzes a collection of reports to identify the underlying
    failure or root cause across the batch.
    """
    kw_model: KeyBERT = request.app.state.kw_model
    nlp: Language = request.app.state.nlp

    filtered_texts: list[str] = []
    for d in data.descriptions:
        doc = nlp(clean_report(d))
        important_words = [
            t.text for t in doc if t.pos_ in ["NOUN", "VERB", "PROPN"] and not t.is_stop
        ]
        filtered_texts.append(" ".join(important_words))

    cleaned_corpus = " ".join(filtered_texts)

    keywords = kw_model.extract_keywords(
        cleaned_corpus,
        candidates=TECHNICAL_CANDIDATES,
        stop_words=EXTENDED_STOP_WORDS,
        use_mmr=True,
        diversity=0.7,
        top_n=5,
    )

    findings: list[RCAIssue] = []
    for word, score in keywords:
        f_score = score[1] if isinstance(score, tuple) else score
        findings.append(
            RCAIssue(
                root_cause=str(word),
                confidence_score=round(f_score, 4),
                relevance="HIGH" if f_score > 0.5 else "MEDIUM",
            )
        )

    return RCASummary(
        total_processed=len(data.descriptions),
        primary_issue=findings[0].root_cause if findings else "Unknown",
        top_findings=findings,
    )
