from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""

    You are a capable assistant trained to analyze and summarize documents.
    Return only valid JSON matching the exact schema below.
                                          
    {format_instructions}
    Analyze the document 

    {document_text}

"""
    
)