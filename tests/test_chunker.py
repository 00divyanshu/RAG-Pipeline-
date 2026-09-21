import unittest
from langchain_core.documents import Document
from src.chunker import split_documents

class TestChunker(unittest.TestCase):
    def test_empty_documents(self):
        """Splitting empty list should return empty list."""
        chunks = split_documents([])
        self.assertEqual(chunks, [])

    def test_chunking_and_metadata_preservation(self):
        """Test splitting long text and verifying chunk bounds and metadata retention."""
        long_text = "Word " * 500  # ~2500 characters
        doc = Document(
            page_content=long_text,
            metadata={"filename": "test.pdf", "page_number": 3}
        )
        chunks = split_documents([doc], chunk_size=500, chunk_overlap=100)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["filename"], "test.pdf")
            self.assertEqual(chunk.metadata["page_number"], 3)
            self.assertLessEqual(len(chunk.page_content), 550)

if __name__ == "__main__":
    unittest.main()

