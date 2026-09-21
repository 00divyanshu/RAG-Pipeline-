import unittest
from langchain_core.documents import Document
from src.evaluator import RAGEvaluator

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = RAGEvaluator()

    def test_citation_precision(self):
        """Test citation precision calculation."""
        doc1 = Document(page_content="Content 1", metadata={"filename": "doc1.pdf", "page_number": 1})
        doc2 = Document(page_content="Content 2", metadata={"filename": "doc2.pdf", "page_number": 2})

        # All citations match retrieved docs
        result_valid = {
            "source_documents": [doc1, doc2],
            "citations": [
                {"filename": "doc1.pdf", "page": 1},
                {"filename": "doc2.pdf", "page": 2},
            ]
        }
        eval_valid = self.evaluator.evaluate_citations(result_valid)
        self.assertEqual(eval_valid["citation_precision"], 1.0)
        self.assertEqual(eval_valid["valid_count"], 2)

        # One hallucinated citation
        result_hallucinated = {
            "source_documents": [doc1],
            "citations": [
                {"filename": "doc1.pdf", "page": 1},
                {"filename": "fake_doc.pdf", "page": 99},
            ]
        }
        eval_hal = self.evaluator.evaluate_citations(result_hallucinated)
        self.assertEqual(eval_hal["citation_precision"], 0.5)
        self.assertEqual(eval_hal["valid_count"], 1)

    def test_lexical_overlap(self):
        """Test keyword overlap between query and retrieved context."""
        docs = [Document(page_content="Employees receive a monthly wellness stipend of $100.")]
        score = self.evaluator.evaluate_retrieval_overlap("wellness stipend", docs)
        self.assertEqual(score, 1.0)

        unrelated_score = self.evaluator.evaluate_retrieval_overlap("rocket trajectory astrophysics", docs)
        self.assertEqual(unrelated_score, 0.0)

if __name__ == "__main__":
    unittest.main()

