import logging
from typing import Dict, Any, List, Optional, Tuple, Iterator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from src import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI assistant specialized in answering questions based on provided reference documents.

Guidelines:
1. Answer the user's question using ONLY the provided context.
2. If the context does not contain enough information to answer truthfully, state clearly: "I cannot find the answer to that in the provided documents." Do not invent or hallucinate information.
3. Be concise, precise, and well-structured.
4. When stating facts, refer to the document source and page if relevant.

---
Context from Documents:
{context}
"""

def format_docs(docs: List[Document]) -> str:
    """Formats retrieved document chunks into a contextual text block with source headers."""
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "Unknown Document")
        page = doc.metadata.get("page_number", doc.metadata.get("page", "Unknown"))
        content = doc.page_content.strip()
        formatted_chunks.append(f"[Chunk {i} | Source: {filename}, Page: {page}]\n{content}")
    return "\n\n".join(formatted_chunks)

def get_citations(docs: List[Document]) -> List[Dict[str, Any]]:
    """Extracts unique source citations and snippets from retrieved documents."""
    citations = []
    seen = set()
    for doc in docs:
        filename = doc.metadata.get("filename", "Unknown Document")
        page = doc.metadata.get("page_number", doc.metadata.get("page", "Unknown"))
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            citations.append({
                "filename": filename,
                "page": page,
                "snippet": snippet,
            })
    return citations

class RAGPipeline:
    def __init__(
        self,
        retriever,
        llm_model: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.retriever = retriever
        active_provider = (provider or config.LLM_PROVIDER).lower()

        if active_provider == "gemini" or config.GOOGLE_API_KEY:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = llm_model or config.GEMINI_LLM_MODEL
            logger.info(f"Using Google Gemini Flash LLM: {model}")
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=config.GOOGLE_API_KEY,
            )
        else:
            from langchain_ollama import ChatOllama
            model = llm_model or config.OLLAMA_LLM_MODEL
            url = base_url or config.OLLAMA_BASE_URL
            logger.info(f"Using Ollama LLM: {model} at {url}")
            self.llm = ChatOllama(
                model=model,
                base_url=url,
                temperature=temperature,
            )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        self.output_parser = StrOutputParser()

        # Internal LCEL chain
        self.generation_chain = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | self.prompt
            | self.llm
            | self.output_parser
        )

    def retrieve(self, question: str) -> Tuple[str, List[Dict[str, Any]], List[Document]]:
        """
        Retrieves relevant document chunks and extracts citations.
        Returns (context_string, citations, raw_docs).
        """
        retrieved_docs = self.retriever.invoke(question)
        if not retrieved_docs:
            return "", [], []
        return format_docs(retrieved_docs), get_citations(retrieved_docs), retrieved_docs

    def stream_response(self, context_str: str, question: str) -> Iterator[str]:
        """
        Yields tokens in real-time for fast streaming in Streamlit.
        """
        for chunk in self.generation_chain.stream({
            "context": context_str,
            "question": question,
        }):
            yield chunk

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Synchronous end-to-end RAG execution for CLI / non-streaming queries.
        """
        context_str, citations, retrieved_docs = self.retrieve(question)
        if not context_str:
            return {
                "question": question,
                "answer": "No relevant documents or information found in the vector database.",
                "citations": [],
                "source_documents": [],
            }

        answer = self.generation_chain.invoke({
            "context": context_str,
            "question": question,
        })

        return {
            "question": question,
            "answer": answer.strip(),
            "citations": citations,
            "source_documents": retrieved_docs,
        }
