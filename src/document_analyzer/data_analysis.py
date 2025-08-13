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

    def analyze_document(self, document_text: str) -> dict:
        """
        Analyzes the document and returns the extracted metadata and summary.
        """
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            self.log.info("Meta data analysis chain initalized.")
            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text()
            })
            self.log.info("Meta data extraction successful.", keys=list(response.keys()))
            return response
        except Exception as e:
            self.log.error("Error analyzing document", {e})
            raise DocumentException("Failed to analyze document", sys)
