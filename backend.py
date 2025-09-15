# backend.py - Fixed version for OpenAI API

import os
import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_text
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from typing_extensions import TypedDict
import base64
from typing import List, Dict, Any, Optional
import re
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def debug_environment_variables():
    """Check if OPENAI_API_KEY is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***masked***"
        print(f"✅ OPENAI_API_KEY: {masked}")
        return True, []
    else:
        print("❌ OPENAI_API_KEY not set")
        return False, ["OPENAI_API_KEY"]

class HybridPDFLoader(BaseLoader):
    """PDF loader that extracts text, tables, and images with OpenAI descriptions"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.llm = None

        self.images_folder = os.path.join(os.path.dirname(file_path), "extracted_images")
        os.makedirs(self.images_folder, exist_ok=True)

        try:
            if os.getenv("OPENAI_API_KEY"):
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                self.llm = ChatOpenAI(model=model, temperature=0)
                print(f"OpenAI model {model} initialized successfully for image descriptions")
            else:
                print("OPENAI_API_KEY not set. Image descriptions will be skipped.")
        except Exception as e:
            print(f"OpenAI setup failed: {e}. Image descriptions will be skipped.")

    def _get_image_description(self, image_bytes: bytes, image_ext: str, page_number: int, img_index: int) -> str:
        """Generate AI description for image using OpenAI"""
        if not self.llm or not image_bytes:
            return "Image extracted successfully."

        try:
            image_b64 = base64.b64encode(image_bytes).decode()

            examples = """
Example 1: page1_image1 (architecture diagram of RAG workflow...)
Example 2: page2_image2 (profit per company bar chart with axes, values, and colors...)
"""

            message = HumanMessage(content=[
                {"type": "text", "text": f"Describe this PDF image briefly and structurally. "
                                         f"If it's a chart, include axes, values, and colors. "
                                         f"Examples:\n{examples}\n\n"
                                         f"Now describe this image as page{page_number}_image{img_index} (...)."},
                {"type": "image_url", "image_url": {"url": f"data:image/{image_ext};base64,{image_b64}"}}
            ])

            response = self.llm.invoke([message])
            description = response.content.strip()
            print(f"Generated description for page{page_number}_image{img_index}: {description[:100]}...")
            return description

        except Exception as llm_e:
            print(f"OpenAI failed for image {img_index} on page {page_number}: {llm_e}")
            return "Image extracted successfully."

    # (rest of your PDF text/table/image extraction logic unchanged...)

# State Definition for LangGraph
class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    context: str
    response: str
    image_references: List[str]
    images_displayed: bool
    metadata: Dict[str, Any]
    step: str
    error: Optional[str]

# (chunk_documents_langchain, create_vector_store_langchain, ImageManager, prompt templates remain unchanged)

class StreamlitRAGChain:
    """LangChain-based RAG optimized for OpenAI"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.vectorstore = None
        self.chain = None
        self.image_manager = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.embedding_model = None
        self.llm = None
        self.initialized = False
        self.initialization_error = None

    def initialize_rag(self):
        try:
            env_ok, missing = debug_environment_variables()
            if not env_ok:
                self.initialization_error = f"Missing vars: {', '.join(missing)}"
                return False, self.initialization_error

            print("Loading PDF...")
            loader = HybridPDFLoader(self.pdf_path)
            docs = loader.load()

            print("Chunking...")
            modified_docs = chunk_documents_langchain(docs)

            print("Creating embeddings...")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="intfloat/e5-large-v2",
                encode_kwargs={"normalize_embeddings": True},
            )

            print("Creating vector store...")
            self.vectorstore = create_vector_store_langchain(modified_docs, self.embedding_model)

            print("Initializing image manager...")
            self.image_manager = ImageManager(docs)

            print("Connecting to OpenAI...")
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.llm = ChatOpenAI(model=model, temperature=0)
            test = self.llm.invoke([HumanMessage(content="Hello")])
            print(f"✅ Connection test: {test.content[:50]}...")

            print("🔄 Building RAG chain...")
            qa_prompt = PromptTemplate(
                template="""Use the following context to answer the question. 
                Convert image refs like [Image: Image_1_Page_1] to [IMAGE:page1_img1].
                
                Context: {context}
                Question: {question}
                
                Answer:""",
                input_variables=["context", "question"]
            )

            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(),
                memory=self.memory,
                return_source_documents=True,
                output_key="answer",
                combine_docs_chain_kwargs={"prompt": qa_prompt}
            )

            self.initialized = True
            print("✅ RAG system initialized.")
            return True, "Success"

        except Exception as e:
            self.initialization_error = str(e)
            print(f"❌ Init failed: {e}")
            return False, self.initialization_error

# Wrapper
def get_rag_engine(pdf_path: str):
    engine = StreamlitRAGChain(pdf_path)
    success, err = engine.initialize_rag()
    return engine if success else None
