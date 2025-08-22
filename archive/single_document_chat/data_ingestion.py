import uuid
from pathlib import Path
import sys
from datetime import datetime, timezone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentException
from utils.model_loader import ModelLoader

class SingleDocIngestor:
    def __init__(self, data_dir:str = "data/single_document_chat", faiss_dir: str = "faiss_index"):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.faiss_dir = Path(faiss_dir)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)
            self.model_loader = ModelLoader()
            self.log.info("SingleDocIngestor initialized successfully.", temp_path = str(self.data_dir), faiss_dir = str(self.faiss_dir))
        except Exception as e:
            self.log.error(f"Error initializing SingleDocIngestor: {e}")
            raise DocumentException(f"Error initializing SingleDocIngestor: {e}", sys)

    def ingest_files(self, uploaded_files):
        try:
            documents = []

            for uploaded_file in uploaded_files:
                unique_filename = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                temp_path = self.data_dir / unique_filename

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                self.log.info(f"PDF saved for ingestion", filename=uploaded_file.name)
                loader = PyPDFLoader(str(temp_path))
                docs = loader.load()
                documents.extend(docs)
            self.log.info("PDF files loaded successfully.", count=len(documents))

            self.log.info("Files ingested successfully.", count=len(documents))
            return self._create_retriever(documents)
        except Exception as e:
            self.log.error(f"Error ingesting files: {e}")
            raise DocumentException(f"Error ingesting files: {e}", sys)

    def _create_retriever(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            chunks = splitter.split_documents(documents)
            self.log.info("Documents split into chunks.", chunks=len(chunks))

            embeddings = self.model_loader.load_embeddings()
            vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
            vector_store.save_local(str(self.faiss_dir))

            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            self.log.info("FAISS vector store created successfully.", retriever_type=str(type(retriever)))
            return retriever
        except Exception as e:
            self.log.error(f"Error creating retriever: {e}")
            raise DocumentException(f"Error creating retriever: {e}", sys)