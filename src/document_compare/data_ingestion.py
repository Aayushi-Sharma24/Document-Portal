import sys
from pathlib import Path
import fitz
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentException
from typing import Optional
import shutil
import uuid

class DocumentIngestion:
    def __init__(self, base_dir: str = "data/document_compare", session_id: Optional[str] = None):
        self.log = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)
        self.session_id = session_id or str(uuid.uuid4())
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.log.info("DocumentComparator initialized", session_path=str(self.session_path))

    def save_uploaded_files(self, reference_file, actual_file):
        """
        Save uploaded files to the specified directory.
        """
        try:
            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name
            for fobj, out in ((reference_file, ref_path), (actual_file, act_path)):
                if not fobj.name.lower().endswith(".pdf"):
                    raise ValueError("Only PDF files are allowed.")
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
            self.log.info("Files saved", reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return ref_path, act_path
        
        except Exception as e:
            self.log.error(f"Error saving files: {e}")
            raise DocumentException(f"Error saving files: {e}", sys)

    def read_pdf(self, pdf_path: Path) -> str:
        """
        Read the PDF file and extracts the text from each page.
        """
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                all_text = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()

                    if text.strip():
                        all_text.append(f"\n----Page {page_num + 1}----\n{text}")
                self.log.info("PDF read successfully.", file = str(pdf_path), pages=len(all_text))
                return "\n".join(all_text)
        except Exception as e:
            self.log.error(f"Error reading PDF: {e}")
            raise DocumentException(f"Error reading PDF: {pdf_path.name}", sys)
        
    def combine_documents(self) -> str:
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content = self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            combined_text = "\n\n".join(doc_parts)
            self.log.info("Documents combined", count=len(doc_parts), session=self.session_id)
            return combined_text
        
        except Exception as e:
            self.log.error(f"Error combining documents: {e}")
            raise DocumentException("Error combining documents", sys)
        
        
    def clean_old_sessions(self, keep_latest: int = 3):
        try:
            sessions = sorted([f for f in self.base_dir.iterdir() if f.is_dir()], reverse=True)
            for folder in sessions[keep_latest:]:
                shutil.rmtree(folder, ignore_errors=True)
                self.log.info("Old session folder deleted", path=str(folder))
        except Exception as e:
            self.log.error("Error cleaning old sessions", error=str(e))
            raise DocumentException("Error cleaning old sessions", e) from e
