import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from exception.custom_exception import DocumentException
from logger.custom_logger import CustomLogger
from utils.model_loader import ModelLoader

class DocumentIngestor:
    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}
    def __init__(self, temp_dir:str = "data/multi_document_chat", faiss_dir:str = "faiss_index", session_id: str | None = None):
        try:
            self.log = CustomLogger().get_logger()

            #base dir
            self.temp_dir = Path(temp_dir)
            self.faiss_dir = Path(faiss_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)

            #sessioned paths
            self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.session_temp_dir = self.temp_dir / self.session_id
            self.session_faiss_dir = self.faiss_dir / self.session_id
            self.session_temp_dir.mkdir(parents=True, exist_ok=True)
            self.session_faiss_dir.mkdir(parents=True, exist_ok=True)

            self.model_loader = ModelLoader()
            self.log.info(
                "DocumentIngestor initialized successfully.",
                temp_base=str(self.temp_dir), 
                faiss_base=str(self.faiss_dir), 
                session_id=self.session_id, 
                temp_path=str(self.session_temp_dir), 
                faiss_path=str(self.session_faiss_dir)
            )

        except Exception as e:
            self.log.error(f"Error initializing DocumentIngestor", error=str(e))
            raise DocumentException("Initialization error in DocumentIngestor", sys)

    def ingest_files(self, uploaded_files):
        try:
            documents = []
            for uploaded_files in uploaded_files:
                ext = Path(uploaded_files.name).suffix.lower()
                if ext not in self.SUPPORTED_EXTENSIONS:
                    self.log.warning("Unsupported file skipped", filename=uploaded_files.name)
                    continue

                unique_filename = f"{uuid.uuid4().hex[:8]}{ext}"
                temp_path = self.session_temp_dir / unique_filename

                with open(temp_path, "wb") as f:
                    f.write(uploaded_files.read())
                
                self.log.info("File saved", filename=unique_filename, saved_as=str(temp_path), session_id=self.session_id)

                if ext == ".pdf":
                    loader = PyPDFLoader(str(temp_path))
                elif ext == ".docx":
                    loader = Docx2txtLoader(str(temp_path))
                elif ext == ".txt":
                    loader = TextLoader(str(temp_path), encoding="utf-8")
                else:
                    self.log.warning("Unsupported file type", filename=uploaded_files.name)
                    continue

                docs = loader.load()
                documents.extend(docs)

            if not docs:
                raise DocumentException("No valid documents loaded.", sys)

            self.log.info("Documents loaded successfully", total_docs=len(docs), session_id=self.session_id)
            return self._create_retriever(documents)

        except Exception as e:
            self.log.error(f"Error ingesting files", error=str(e))
            raise DocumentException("Ingestion error in DocumentIngestor", sys)

    def _create_retriever(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            chunks = splitter.split_documents(documents)
            self.log.info("Documents split into chunks", total_chunks=len(chunks), session_id=self.session_id)
            embeddings = self.model_loader.load_embeddings()
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

            # Save FAISS index under session folder
            vectorstore.save_local(str(self.session_faiss_dir))
            self.log.info("FAISS index created and saved", session_id=self.session_id, faiss_path=str(self.session_faiss_dir))

            retriever = vectorstore.as_retriever(search_type ="similarity", search_kwargs={"k": 5})
            self.log.info("Retriever created successfully", session_id=self.session_id)
            return retriever
        
        except Exception as e:
            self.log.error(f"Error creating retriever", error=str(e))
            raise DocumentException("Retriever creation error in DocumentIngestor", sys)
