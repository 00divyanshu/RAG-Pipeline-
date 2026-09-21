import logging
from typing import Dict, Any, List, Optional, Tuple, Iterator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from src import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI knowledge assistant specialized in explaining and answering questions from reference documents.

Guidelines:
1. Answer the user's question clearly and accurately using the provided context from the documents.
2. If the user asks to read out, explain, or summarize the document, provide a comprehensive summary of its key points based on the context.
3. When stating facts, refer to the document source and page number if available.
4. If the context does not contain enough information to answer a specific question, clarify politely what is covered in the documents rather than giving a flat refusal. Do not invent facts not present in the context.

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
        google_key = config.GOOGLE_API_KEY

        if active_provider == "gemini" or google_key:
            if not google_key:
                raise ValueError(
                    "Google Gemini API Key is missing! "
                    "Please configure GOOGLE_API_KEY in your Streamlit Cloud Secrets or via the sidebar."
                )
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = llm_model or config.GEMINI_LLM_MODEL
            logger.info(f"Using Google Gemini Flash LLM: {model}")
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=google_key,
            )
        else:
            if config.is_cloud_environment():
                raise ValueError(
                    "Running in the cloud (Streamlit Community Cloud), but GOOGLE_API_KEY is not configured! "
                    "Local Ollama is not available in cloud environments. Please enter your Google API Key in the sidebar or Streamlit Secrets."
                )
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
        retrieved_docs = []
        try:
            retrieved_docs = self.retriever.invoke(question)
        except Exception as e:
            logger.warning(f"Initial retrieval error: {e}")

        if not retrieved_docs:
            # Fallback for broad or conversational overview queries
            general_keywords = ["read", "read out", "tell me", "what is this", "summarize", "summary", "overview", "what is in", "what does", "about", "content", "document", "file", "explain"]
            q_lower = question.lower()
            if any(kw in q_lower for kw in general_keywords) or len(question.strip().split()) <= 4:
                try:
                    logger.info("Direct query returned no results; running broad fallback retrieval...")
                    fallback_docs = self.retriever.invoke("document overview summary main content")
                    if fallback_docs:
                        retrieved_docs = fallback_docs
                except Exception as e:
                    logger.warning(f"Fallback retrieval error: {e}")

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
