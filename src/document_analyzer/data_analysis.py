import os
import sys
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import *

class DocumentAnalyzer:
    """
    Analyses document using pretrained models.
    Automatically logs all actions and supports session-based organisations.
    """

    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()

            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

            self.prompt = prompt
            self.log.info("Document Analyzer initialized successfully")

        except Exception as e:
            self.log.error("Error initializing DocumentAnalyzer", {e})
            raise DocumentException("Failed to initialize DocumentAnalyzer", sys) 

    def analyze_document(self):
        pass

