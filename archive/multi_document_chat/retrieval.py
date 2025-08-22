import sys
import os
from typing import Optional
from langchain_core.messages import BaseMessage
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType

class ConversationalRAG:
    def __init__(self, session_id: str, retriever=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            if retriever is None:
                raise ValueError("Retriever cannot be none")
            
            self.retriever = retriever
            self._build_lcel_chain()
            self.log.info("ConversationalRAG initialized successfully", session_id=self.session_id)
        except Exception as e:
            self.log.error("Error initializing ConversationalRAG", error=str(e))
            raise DocumentException("Error initializing ConversationalRAG",sys) 

    def load_retriever_from_faiss(self, index_path:str):
        """
        Load a FAISS vectorestore from disk and covert to retriever.
        """
        try:
            embeddings = ModelLoader().load_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS Index path {index_path} does not exist.")
            
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

            self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            self.log.info("Retriever loaded from FAISS index", index_path=index_path, session_id=self.session_id)
            return self.retriever
        except Exception as e:
            self.log.error("Error loading retriever from FAISS", error=str(e))
            raise DocumentException("Error loading retriever from FAISS", sys)

    def invoke(self, user_input: str, chat_history: Optional[list[BaseMessage]] = None) -> str:
        try:
            chat_history = chat_history or []
            payload = {
                "input": user_input,
                "chat_history": chat_history
            }
            answer = self.chain.invoke(payload)
            if not answer:
                self.log.warning("No answer generated", session_id=self.session_id, user_input=user_input)

            self.log.info("Chain invoked successfully", session_id=self.session_id, user_input=user_input, answer_preview=answer[:150])
            return answer
        except Exception as e:
            self.log.error("Error invoking ConversationalRAG", error=str(e))
            raise DocumentException("Error invoking ConversationalRAG", sys)


    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            self.log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            self.log.error("Error loading LLM", error=str(e))
            raise DocumentException("Error loading LLM", sys)

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def _build_lcel_chain(self):
        try:
            # 1. Rewrite user query using chat history
            question_rewriter = (
                {
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            
            )

            # 2. Retrieve docs for rewritten query
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3. Feed Context + original input + chat history into answer prompt
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )
        except Exception as e:
            self.log.error("Error building LCEL chain", error=str(e))
            raise DocumentException("Error building LCEL chain", sys)
