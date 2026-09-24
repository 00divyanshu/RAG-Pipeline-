import unittest
from pathlib import Path
from src.loader import (
    load_documents_from_directory,
    load_single_pdf,
    load_single_document,
    load_single_csv,
    load_single_text,
    load_single_docx,
)

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

    def test_csv_loading(self):
        """Test loading and structured formatting of a CSV file."""
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("Department,Budget,Lead\nEngineering,500000,Sarah\nMarketing,200000,Dave\n")
            temp_path = Path(f.name)

        try:
            docs = load_single_csv(temp_path)
            self.assertGreater(len(docs), 0)
            self.assertIn("Engineering", docs[0].page_content)
            self.assertIn("Sarah", docs[0].page_content)
            self.assertEqual(docs[0].metadata["file_type"], "csv")
        finally:
            temp_path.unlink()

    def test_text_and_markdown_loading(self):
        """Test loading plain text and markdown documents."""
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Architecture Overview\nThis is a multi-tenant cloud RAG system.\n")
            temp_path = Path(f.name)

        try:
            docs = load_single_document(temp_path)
            self.assertEqual(len(docs), 1)
            self.assertIn("Architecture Overview", docs[0].page_content)
            self.assertEqual(docs[0].metadata["file_type"], "md")
        finally:
            temp_path.unlink()

    def test_docx_loading(self):
        """Test creating and loading a Microsoft Word docx file."""
        import tempfile
        import docx
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            temp_path = Path(f.name)

        try:
            doc = docx.Document()
            doc.add_heading("Project Whitepaper", 0)
            doc.add_paragraph("This is the executive summary of our AI knowledge base.")
            doc.save(str(temp_path))

            loaded_docs = load_single_docx(temp_path)
            self.assertGreater(len(loaded_docs), 0)
            self.assertIn("Project Whitepaper", loaded_docs[0].page_content)
            self.assertEqual(loaded_docs[0].metadata["file_type"], "docx")
        finally:
            temp_path.unlink()

if __name__ == "__main__":
    unittest.main()

