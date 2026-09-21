import unittest
from pathlib import Path
from src.loader import load_documents_from_directory, load_single_pdf

class TestLoader(unittest.TestCase):
    def test_empty_directory(self):
        """Test loader on an empty or non-existent temporary directory."""
        temp_dir = Path("data/non_existent_test_dir")
        docs = load_documents_from_directory(temp_dir)
        self.assertEqual(docs, [])
        if temp_dir.exists():
            temp_dir.rmdir()

    def test_sample_pdf_loading(self):
        """Test loading existing sample PDF in data/docs."""
        sample_path = Path("data/docs/sample_company_policy.pdf")
        if sample_path.exists():
            docs = load_single_pdf(sample_path)
            self.assertGreater(len(docs), 0)
            self.assertEqual(docs[0].metadata.get("filename"), "sample_company_policy.pdf")
            self.assertEqual(docs[0].metadata.get("page_number"), 1)
            self.assertIn("Acme Corp", docs[0].page_content)

if __name__ == "__main__":
    unittest.main()

