import unittest
from langchain_core.documents import Document
from src.rag_chain import format_docs, get_citations

class TestRAGChainFormatting(unittest.TestCase):
    def test_format_docs(self):
        """Test formatting of chunks into structured text blocks."""
        docs = [
            Document(page_content="Policy 1 details", metadata={"filename": "docA.pdf", "page_number": 2}),
            Document(page_content="Policy 2 details", metadata={"filename": "docB.pdf", "page_number": 5}),
        ]
        formatted = format_docs(docs)
        self.assertIn("[Chunk 1 | Source: docA.pdf, Page: 2]", formatted)
        self.assertIn("Policy 1 details", formatted)
        self.assertIn("[Chunk 2 | Source: docB.pdf, Page: 5]", formatted)
        self.assertIn("Policy 2 details", formatted)

    def test_get_citations(self):
        """Test citation extraction and deduplication."""
        docs = [
            Document(page_content="First instance", metadata={"filename": "handbook.pdf", "page_number": 1}),
            Document(page_content="Second instance same page", metadata={"filename": "handbook.pdf", "page_number": 1}),
            Document(page_content="New page instance", metadata={"filename": "handbook.pdf", "page_number": 2}),
        ]
        citations = get_citations(docs)
        # Should deduplicate page 1
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["filename"], "handbook.pdf")
        self.assertEqual(citations[0]["page"], 1)
        self.assertEqual(citations[1]["page"], 2)

if __name__ == "__main__":
    unittest.main()

