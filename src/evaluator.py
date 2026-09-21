import re
import logging
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

EVALUATION_PROMPT = """You are an impartial AI evaluator assessing the quality and faithfulness of an answer produced by a Retrieval-Augmented Generation (RAG) system.

Evaluate the following generation based on:
1. Faithfulness (Is the answer strictly supported by the retrieved context without hallucination?)
2. Context Relevance (Did the retrieved context contain the information needed to answer the question?)
3. Answer Relevance (Does the answer directly address the user's question?)

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Output your evaluation in the following strict JSON format:
{{
  "faithfulness_score": <number between 0.0 and 1.0>,
  "context_relevance_score": <number between 0.0 and 1.0>,
  "answer_relevance_score": <number between 0.0 and 1.0>,
  "verdict": "<PASS or FAIL>",
  "reasoning": "<short explanation of the scores>"
}}
"""

class RAGEvaluator:
    """Evaluator for assessing the quality, faithfulness, and relevance of RAG responses."""

    def __init__(self, llm=None):
        self.llm = llm
        if self.llm:
            self.eval_prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT)
            self.eval_chain = self.eval_prompt | self.llm | StrOutputParser()

    def evaluate_citations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validates that all citations correspond to actual retrieved documents."""
        source_docs = result.get("source_documents", [])
        citations = result.get("citations", [])

        if not citations and not source_docs:
            return {"citation_precision": 1.0, "valid_count": 0, "total_count": 0}

        retrieved_sources = {
            (doc.metadata.get("filename"), doc.metadata.get("page_number", doc.metadata.get("page")))
            for doc in source_docs
        }

        valid_citations = 0
        for cite in citations:
            key = (cite.get("filename"), cite.get("page"))
            if key in retrieved_sources:
                valid_citations += 1

        precision = valid_citations / len(citations) if citations else 1.0
        return {
            "citation_precision": round(precision, 2),
            "valid_count": valid_citations,
            "total_count": len(citations),
        }

    def evaluate_retrieval_overlap(self, question: str, retrieved_docs: List[Any]) -> float:
        """Heuristic check: calculates keyword overlap between query and retrieved context."""
        if not retrieved_docs:
            return 0.0

        query_terms = set(re.findall(r"\w+", question.lower()))
        # Filter common stopwords
        stopwords = {"what", "is", "the", "and", "or", "to", "in", "of", "how", "many", "do", "does", "a", "an", "can"}
        query_terms = {t for t in query_terms if t not in stopwords and len(t) > 2}

        if not query_terms:
            return 1.0

        all_content = " ".join([d.page_content.lower() for d in retrieved_docs])
        found_terms = [t for t in query_terms if t in all_content]

        return round(len(found_terms) / len(query_terms), 2)

    def evaluate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation on a RAG result dictionary.
        Returns citation validity, lexical overlap score, and (if LLM is available) LLM-as-a-judge scores.
        """
        question = result.get("question", "")
        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])
        citations = result.get("citations", [])

        citation_eval = self.evaluate_citations(result)
        lexical_overlap = self.evaluate_retrieval_overlap(question, source_docs)

        eval_report = {
            "citation_precision": citation_eval["citation_precision"],
            "lexical_overlap_score": lexical_overlap,
            "retrieved_chunk_count": len(source_docs),
            "citations_provided": len(citations),
        }

        if self.llm and source_docs:
            try:
                context_str = "\n\n".join([d.page_content for d in source_docs])
                judge_output = self.eval_chain.invoke({
                    "question": question,
                    "context": context_str,
                    "answer": answer,
                })
                eval_report["llm_evaluation"] = judge_output.strip()
            except Exception as e:
                logger.warning(f"LLM judge evaluation failed: {e}")
                eval_report["llm_evaluation_error"] = str(e)

        return eval_report

