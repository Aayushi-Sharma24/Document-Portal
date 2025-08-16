import sys, os
from dotenv import load_dotenv
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentException
from utils.model_loader import ModelLoader
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from utils.model_loader import ModelLoader
import streamlit as st

load_dotenv()


class ConversationalRAG:
    def __init__(self, session_id: str, retriever):
        self.log = CustomLogger().get_logger(__name__)
        self.session_id = session_id
        self.retriever = retriever
        self.llm = self._load_llm()
        self.contextualize_prompt = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
        self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
        self.history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, self.contextualize_prompt)
        self.log.info("created history aware retriever", session_id=session_id)
        self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)
        self.log.info("created rag chain", session_id=session_id)
        self.chain = RunnableWithMessageHistory(
            self.rag_chain,
            self._get_session_history,
            input_message_key="input",
            history_message_key="chat_history",
            output_message_key="answer"
        )
        self.log.info("ConversationalRAG initialized", session_id=session_id)


    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            self.log.info("LLM loaded successfully", class_name=llm.__class__.__name__)
            return llm
        except Exception as e:
            self.log.error(f"Error loading LLM: {e}")
            raise DocumentException(f"Error loading LLM: {e}", sys)

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        try:
            if not hasattr(self, "_store"):
                self._store = {}
            if session_id not in self._store:
                self._store[session_id] = ChatMessageHistory()
                self.log.info("Created new session history", session_id=session_id)
            return self._store[session_id]
        except Exception as e:
            self.log.error(f"Error loading session history: {e}")
            raise DocumentException(f"Error loading session history: {e}", sys)
        
    def load_retriever_from_faiss(self, index_path):
        try:
            embeddings = ModelLoader().get_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")
        
            vectorstore = FAISS.load_local(index_path, embeddings)
            self.log.info("FAISS vector store loaded successfully.", index_path=index_path)
            return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        except Exception as e:
            self.log.error(f"Error loading FAISS vector store: {e}")
            raise DocumentException(f"Error loading FAISS vector store: {e}", sys)
        
    def invoke(self, user_input:str)->str:
        try:
            response = self.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}}
            )
            answer = response.get("answer", "No answer")
            if not answer:
                self.log.warning("No answer found in the response.", session_id=self.session_id)
            
            self.log.info("RAG chain invoked successfully", session_id=self.session_id, user_input=user_input, answer=answer[:150])
            return answer
        except Exception as e:
            self.log.error(f"Error invoking RAG chain: {e}", session_id=self.session_id)
            return "Error invoking RAG chain."
