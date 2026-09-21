import unittest
from unittest.mock import MagicMock
from pathlib import Path
from src.vectorstore import delete_document_by_name, list_indexed_documents

class TestVectorStoreManagement(unittest.TestCase):
    def test_list_indexed_documents_chroma(self):
        """Test listing documents and counting chunks from Chroma metadata."""
        mock_vs = MagicMock()
        mock_vs.get.return_value = {
            "metadatas": [
                {"filename": "docA.pdf"},
                {"filename": "docA.pdf"},
                {"filename": "docB.pdf"},
            ]
        }
        docs = list_indexed_documents(mock_vs)
        self.assertEqual(len(docs), 2)
        docA = next(d for d in docs if d["filename"] == "docA.pdf")
        docB = next(d for d in docs if d["filename"] == "docB.pdf")
        self.assertEqual(docA["chunks"], 2)
        self.assertEqual(docB["chunks"], 1)

    def test_delete_document_by_name_chroma(self):
        """Test deleting document chunks by filename metadata."""
        mock_vs = MagicMock()
        mock_vs.get.return_value = {"ids": ["id-1", "id-2"]}
        
        result = delete_document_by_name(mock_vs, "docA.pdf")
        self.assertTrue(result)
        mock_vs.get.assert_called_once_with(where={"filename": "docA.pdf"})
        mock_vs.delete.assert_called_once_with(ids=["id-1", "id-2"])

if __name__ == "__main__":
    unittest.main()
