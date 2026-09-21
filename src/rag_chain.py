import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

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
    def __init__(self, retriever, llm_model: str, base_url: str, temperature: float = 0.2):
        self.retriever = retriever
        self.llm = ChatOllama(
            model=llm_model,
            base_url=base_url,
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

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Executes end-to-end RAG:
        1. Retrieves relevant chunks from vector store.
        2. Formats chunks with metadata.
        3. Generates response using LLM.
        4. Returns answer along with structured citations.
        """
        retrieved_docs = self.retriever.invoke(question)
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "No relevant documents or information found in the vector database.",
                "citations": [],
                "source_documents": [],
            }

        context_str = format_docs(retrieved_docs)
        answer = self.generation_chain.invoke({
            "context": context_str,
            "question": question,
        })
        citations = get_citations(retrieved_docs)

        return {
            "question": question,
            "answer": answer.strip(),
            "citations": citations,
            "source_documents": retrieved_docs,
        }

