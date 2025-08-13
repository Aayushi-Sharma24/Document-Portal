import os
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

class DocumentAnalyzer:
    """
    Analyses document using pretrained models.
    Automatically logs all actions and supports session-based organisations.
    """

    def __init__(self):
        pass

    def analyze_document(self):
        pass

