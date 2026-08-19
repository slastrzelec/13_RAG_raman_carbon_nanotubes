"""
RAGAs evaluation script (Phase 0, final step).

Runs the existing RAG pipeline (retrieval + generation) against a fixed set of
evaluation questions, then scores the answers using RAGAs' faithfulness metric:
does the generated answer stick to facts present in the retrieved context, or
does it hallucinate claims not supported by the sources?

Note: RAGAs' context_precision metric was intentionally left out here, since it
requires a "reference" (ground-truth) answer per question, which this eval set
does not include (Option A from the phase-0 planning: questions only, no manual
reference answers). Could be added later as an extension.

This is a standalone script, not a pytest test — it makes real calls to the OpenAI
API (both for answer generation and for RAGAs' LLM-as-judge scoring), so it costs
money and is slow. Run it manually, not as part of CI.

Usage:
    python -m src.evaluation
"""
import json
import logging
import os
import sys
import types

from dotenv import load_dotenv
from datasets import Dataset

load_dotenv()  # must run before any module that creates an OpenAI client at import time

# Workaround: some ragas versions import langchain_community.chat_models.vertexai,
# a module that was removed from langchain_community during its deprecation/sunset.
# We don't use Vertex AI, so we stub the module out before ragas tries to import it.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    stub = types.ModuleType("langchain_community.chat_models.vertexai")
    stub.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = stub

from ragas import evaluate
from ragas.metrics import faithfulness
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from src import config
from src.retrieval import Retriever
from src.generation import rag_query
from src.eval_dataset import EVAL_QUESTIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(config.BASE_DIR, "data", "evaluation")
RESULTS_PATH = os.path.join(RESULTS_DIR, "ragas_results.json")


def collect_rag_outputs(retriever: Retriever, questions: list[str]) -> dict:
    """
    Runs the real RAG pipeline for each question and collects what RAGAs needs:
    question, generated answer, and the retrieved context (as plain strings).
    """
    records = {"question": [], "answer": [], "contexts": []}

    for i, question in enumerate(questions, start=1):
        logger.info(f"[{i}/{len(questions)}] Running RAG for: {question}")
        answer, retrieved_chunks = rag_query(retriever, question)

        records["question"].append(question)
        records["answer"].append(answer)
        records["contexts"].append([chunk["text"] for chunk in retrieved_chunks])

    return records


def run_evaluation():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("Loading retriever...")
    retriever = Retriever()

    logger.info(f"Collecting RAG outputs for {len(EVAL_QUESTIONS)} questions...")
    records = collect_rag_outputs(retriever, EVAL_QUESTIONS)
    dataset = Dataset.from_dict(records)

    # RAGAs uses an LLM as a judge — reuse the same model as the app for consistency and cost
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=config.LLM_MODEL))

    logger.info("Running RAGAs evaluation (faithfulness)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=judge_llm,
    )

    result_df = result.to_pandas()

    per_question = result_df[["user_input", "faithfulness"]].rename(
        columns={"user_input": "question"}
    ).to_dict(orient="records")

    # Attach full answer + retrieved context per question — needed to debug low-scoring cases,
    # since the faithfulness number alone doesn't explain *why* it's low.
    contexts_by_question = dict(zip(records["question"], records["contexts"]))
    answers_by_question = dict(zip(records["question"], records["answer"]))
    for row in per_question:
        row["answer"] = answers_by_question.get(row["question"], "")
        row["retrieved_context"] = contexts_by_question.get(row["question"], [])
    summary = {
        "mean_faithfulness": round(float(result_df["faithfulness"].mean()), 4),
        "num_questions": len(EVAL_QUESTIONS),
    }

    output = {"summary": summary, "per_question": per_question}

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Mean faithfulness: {summary['mean_faithfulness']}")
    logger.info(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()