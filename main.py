import sys
import argparse
import logging
from pathlib import Path
from src import config
from src.loader import load_documents_from_directory
from src.chunker import split_documents
from src.vectorstore import (
    get_embeddings,
    get_vector_store,
    index_documents,
    get_retriever,
)
from src.rag_chain import RAGPipeline
from src.evaluator import RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def ingest_cmd(args):
    """Ingests all supported documents from data/docs and indexes them."""
    namespace = getattr(args, "namespace", "default")
    backend_name = "Pinecone Serverless" if config.PINECONE_API_KEY else f"ChromaDB ({config.CHROMA_PERSIST_DIR})"
    print("=" * 60)
    print(" Starting Document Ingestion Pipeline")
    print("=" * 60)
    print(f" Source Directory    : {config.DOCS_DIR}")
    print(f" Vector Backend      : {backend_name}")
    print(f" Tenant Vault / NS   : {namespace}")
    print(f" Embedding Model     : {config.EMBEDDING_MODEL}")
    print(f" Chunk Size / Overlap: {config.CHUNK_SIZE} / {config.CHUNK_OVERLAP}")
    print("=" * 60)

    docs = load_documents_from_directory(config.DOCS_DIR)
    if not docs:
        print(f"\n[!] No documents found in '{config.DOCS_DIR}'.")
        print("    Please drop supported files (.pdf, .docx, .csv, .txt, .md) into that folder and re-run ingest.\n")
        return

    chunks = split_documents(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    embeddings = get_embeddings()

    index_documents(
        documents=chunks,
        persist_directory=config.CHROMA_PERSIST_DIR,
        embeddings=embeddings,
        recreate=args.recreate,
        namespace=namespace,
    )

    print(f"\n[+] Ingestion complete! Indexed {len(chunks)} chunks into vault '{namespace}'.\n")

def get_pipeline(namespace: str = "default"):
    """Builds and returns the RAG pipeline."""
    embeddings = get_embeddings()
    vector_store = get_vector_store(config.CHROMA_PERSIST_DIR, embeddings, namespace=namespace)
    retriever = get_retriever(vector_store, search_type="similarity", k=config.RETRIEVER_K)
    return RAGPipeline(
        retriever=retriever,
        llm_model=config.LLM_MODEL,
    )

def print_result(result: dict):
    """Pretty prints the question, answer, and citations."""
    print("\n" + "=" * 60)
    print(f"Q: {result['question']}")
    print("-" * 60)
    print(f"A:\n{result['answer']}")
    print("-" * 60)
    if result["citations"]:
        print("Citations & Sources:")
        for i, cite in enumerate(result["citations"], 1):
            print(f"  [{i}] {cite['filename']} (Page {cite['page']})")
            print(f"      \"{cite['snippet']}\"")
    else:
        print("No source citations found.")
    print("=" * 60 + "\n")

def query_cmd(args):
    """Executes a single question."""
    namespace = getattr(args, "namespace", "default")
    pipeline = get_pipeline(namespace=namespace)
    provider_name = "Groq LPU" if config.LLM_PROVIDER == "groq" else "Google Gemini"
    print(f"\nProcessing query with {provider_name} ({config.LLM_MODEL}) in vault '{namespace}'...")
    result = pipeline.ask(args.question)
    print_result(result)

def chat_cmd(args):
    """Starts an interactive Q&A session in the terminal."""
    namespace = getattr(args, "namespace", "default")
    pipeline = get_pipeline(namespace=namespace)
    provider_name = "Groq LPU" if config.LLM_PROVIDER == "groq" else "Google Gemini"
    print("=" * 60)
    print(" Interactive RAG Chat Session")
    print(f" Engine: {provider_name} ({config.LLM_MODEL}) | Embeddings: {config.EMBEDDING_MODEL}")
    print(f" Target Vault: {namespace}")
    print(" Type 'exit', 'quit', or 'q' to end the session.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break
            result = pipeline.ask(question)
            print_result(result)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

def evaluate_cmd(args):
    """Executes a question and runs quality evaluation metrics on the result."""
    namespace = getattr(args, "namespace", "default")
    pipeline = get_pipeline(namespace=namespace)
    provider_name = "Groq LPU" if config.LLM_PROVIDER == "groq" else "Google Gemini"
    print(f"\nProcessing query with {provider_name} ({config.LLM_MODEL}) in vault '{namespace}'...")
    result = pipeline.ask(args.question)
    print_result(result)

    print("-" * 60)
    print(" Running Quality & Faithfulness Evaluation...")
    print("-" * 60)
    evaluator = RAGEvaluator(llm=pipeline.llm if args.with_judge else None)
    eval_report = evaluator.evaluate(result)

    print(f"  Citation Precision     : {eval_report['citation_precision'] * 100:.1f}%")
    print(f"  Lexical Overlap Score  : {eval_report['lexical_overlap_score'] * 100:.1f}%")
    print(f"  Retrieved Chunks Count : {eval_report['retrieved_chunk_count']}")
    print(f"  Citations Count        : {eval_report['citations_provided']}")

    if "llm_evaluation" in eval_report:
        print("\nLLM-as-a-Judge Report:\n", eval_report["llm_evaluation"])
    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Enterprise AI Assistant & Document Intelligence CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Tenant vault namespace to target (default: 'default')",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Index documents from data/docs into vector storage")
    ingest_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Wipe and rebuild the vector database from scratch",
    )
    ingest_parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Tenant vault namespace to index into (default: 'default')",
    )
    ingest_parser.set_defaults(func=ingest_cmd)

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a single question")
    query_parser.add_argument("question", type=str, help="Question to ask the knowledge base")
    query_parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Tenant vault namespace to query (default: 'default')",
    )
    query_parser.set_defaults(func=query_cmd)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Ask a question and evaluate retrieval & faithfulness")
    eval_parser.add_argument("question", type=str, help="Question to evaluate against the knowledge base")
    eval_parser.add_argument(
        "--with-judge",
        action="store_true",
        help="Use LLM-as-a-judge to score answer faithfulness and relevance",
    )
    eval_parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Tenant vault namespace to evaluate (default: 'default')",
    )
    eval_parser.set_defaults(func=evaluate_cmd)

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Tenant vault namespace to chat within (default: 'default')",
    )
    chat_parser.set_defaults(func=chat_cmd)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)

if __name__ == "__main__":
    main()

