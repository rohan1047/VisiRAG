# backend.py - Fixed version without Streamlit context issues

import os
import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_text
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.callbacks import StdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import BaseMessage
from langchain.chains import ConversationalRetrievalChain

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
import base64
from typing import List, Dict, Any, Optional, Annotated
import re
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, HTML, Markdown
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import operator
# Remove streamlit import from backend - it will be imported in frontend only
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def debug_environment_variables():
    """Debug function to check environment variables - NO Streamlit functions"""
    required_vars = [
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_KEY", 
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_ENDPOINT"
    ]
    
    print("\n=== Environment Variables Debug ===")
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "KEY" in var:
                masked_value = f"{value[:8]}...{value[-8:]}" if len(value) > 16 else "***masked***"
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET")
            missing_vars.append(var)
    print("=====================================\n")
    
    return len(missing_vars) == 0, missing_vars

class HybridPDFLoader(BaseLoader):
    """Single-class PDF loader with smart library combination"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.azure_llm = None

        # Create extracted images folder
        self.images_folder = os.path.join(os.path.dirname(file_path), "extracted_images")
        os.makedirs(self.images_folder, exist_ok=True)

        try:
            # Check if environment variables are set before initializing AzureChatOpenAI
            if all([os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    os.getenv("AZURE_OPENAI_API_KEY"),
                    os.getenv("AZURE_OPENAI_API_VERSION"),
                    os.getenv("AZURE_OPENAI_ENDPOINT")]):
                self.azure_llm = AzureChatOpenAI(
                    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    temperature=0
                )
                print("Azure OpenAI initialized successfully for image descriptions")
            else:
                print("Azure OpenAI environment variables not fully set. Image descriptions will be skipped.")
        except Exception as e:
            print(f"Azure OpenAI setup failed: {e}. Image descriptions will be skipped.")

    def _get_image_description(self, image_bytes: bytes, image_ext: str, page_number: int, img_index: int) -> str:
        """Generate AI description for image using Azure OpenAI"""
        if not self.azure_llm or not image_bytes:
            return "Image extracted successfully."

        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_bytes).decode()

            # Few-shot examples for guiding the model
            examples = """
Example 1: page1_image1 (architecture of rag with sequential (parallel(input->chunking->embedding->vector store, user input->embedding), retrieval->LLM->output))

Example 2: page2_image2 (profit per company bar chart
company | profit
company1 | 100 units
company2 | 200 units

red color represents company1
blue color represents company2

x-label = company name
y-label = profit)
"""

            message = HumanMessage(content=[
                {"type": "text", "text": f"You are given an image extracted from a PDF. "
                                        f"Write a short structured summary. "
                                        f"If the image is a graph or chart, include axis labels, values, and colors. "
                                        f"Follow the examples below:\n\n{examples}\n\n"
                                        f"Now describe this image as page{page_number}_image{img_index} (...)."},
                {"type": "image_url", "image_url": {"url": f"data:image/{image_ext};base64,{image_b64}"}}
            ])

            response = self.azure_llm.invoke([message])
            description = response.content.strip()
            print(f"Generated description for page{page_number}_image{img_index}: {description[:100]}...")
            return description

        except Exception as llm_e:
            print(f"Azure OpenAI failed for image {img_index} on page {page_number}: {llm_e}")
            return "Image extracted successfully."

    def load(self) -> List[Document]:
        """Single method that extracts text, tables, and images with smart library combination"""
        documents = []

        try:
            # Open with all three libraries simultaneously for efficiency
            doc = fitz.open(self.file_path)  # PyMuPDF for images & page count
            plumber_doc = pdfplumber.open(self.file_path)  # pdfplumber for tables & coordinates

            num_pages = len(doc)
            print(f"Processing PDF with {num_pages} pages")

            for page_num in range(num_pages):
                page_number = page_num + 1
                content_parts = []

                # Initialize comprehensive metadata
                metadata = {
                    "source": self.file_path,
                    "page_number": page_number,
                    "page_index": page_num,
                    "total_pages": num_pages,
                    "document_index": page_num,
                    "elements": {
                        "text": [],
                        "tables": [],
                        "images": []
                    }
                }

                # TEXT LOADER - Use pdfminer.six + pdfplumber coordinates
                try:
                    # Get clean text from pdfminer
                    text = extract_text(self.file_path, page_numbers=[page_num])
                    if text and text.strip():
                        content_parts.append(text.strip())

                    # Get word coordinates from pdfplumber
                    page = plumber_doc.pages[page_num]
                    words = page.extract_words()

                    # Add text metadata with indexing
                    for i, word in enumerate(words):
                        metadata["elements"]["text"].append({
                            "index": i,
                            "text": word['text'],
                            "bbox": (word['x0'], word['top'], word['x1'], word['bottom']),
                            "font": word.get('fontname', 'Unknown'),
                            "size": word.get('size', 0)
                        })

                except Exception as e:
                    print(f"Text extraction error page {page_number}: {e}")

                # TABLE LOADER - Use pdfplumber
                try:
                    page = plumber_doc.pages[page_num]
                    tables = page.extract_tables()

                    for i, table in enumerate(tables):
                        if table:
                            # Clean table data
                            cleaned_table = [[str(cell) if cell is not None else "" for cell in row] for row in table]
                            table_str = '\n'.join([' | '.join(row) for row in cleaned_table])

                            # Get table coordinates
                            table_bbox = None
                            try:
                                found_tables = page.find_tables()
                                if i < len(found_tables):
                                    table_bbox = found_tables[i].bbox
                            except:
                                pass

                            # Add to content and metadata
                            content_parts.append(f"\n[Table {i+1}]\n{table_str}")
                            metadata["elements"]["tables"].append({
                                "index": i,
                                "content": cleaned_table,
                                "bbox": table_bbox,
                                "rows": len(cleaned_table),
                                "cols": len(cleaned_table[0]) if cleaned_table else 0
                            })

                except Exception as e:
                    print(f"Table extraction error page {page_number}: {e}")

                # IMAGE LOADER - Use PyMuPDF with Azure OpenAI descriptions
                try:
                    page_obj = doc[page_num]
                    images = page_obj.get_images(full=True)
                    print(f"Page {page_number}: Found {len(images)} images")

                    for img_index, img in enumerate(images, start=1):
                        try:
                            xref = img[0]  # image reference
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]

                            # Get image coordinates
                            img_rects = page_obj.get_image_rects(xref)
                            bbox = tuple(img_rects[0]) if img_rects else None

                            # Build filename
                            image_filename = f"page{page_number}_img{img_index}.{image_ext}"
                            image_path = os.path.join(self.images_folder, image_filename)

                            # Save image
                            with open(image_path, "wb") as f:
                                f.write(image_bytes)

                            print(f"Saved: {image_filename}")

                            # Generate AI image description using Azure OpenAI
                            image_name = f"Image_{img_index}_Page_{page_number}"
                            image_description = self._get_image_description(image_bytes, image_ext, page_number, img_index)

                            # Add to content in the exact format you specified
                            img_content = f"\n\n[Image: {image_name}]\nDescription: {image_description}"
                            content_parts.append(img_content)

                            # Add to metadata
                            metadata["elements"]["images"].append({
                                "index": img_index - 1,  # 0-based index for consistency
                                "name": image_name,
                                "description": image_description,
                                "bbox": bbox,
                                "format": image_ext,
                                "size_bytes": len(image_bytes),
                                "saved_path": image_path,
                                "filename": image_filename,
                                "xref": xref
                            })

                            print(f"Added image content for {image_name}")

                        except Exception as img_e:
                            print(f"Image processing error {img_index} on page {page_number}: {img_e}")

                except Exception as e:
                    print(f"Image extraction error page {page_number}: {e}")

                # Create final document with all content
                combined_content = "\n".join(content_parts).strip()
                if combined_content:
                    # Add summary counts to metadata
                    metadata.update({
                        "text_count": len(metadata["elements"]["text"]),
                        "table_count": len(metadata["elements"]["tables"]),
                        "image_count": len(metadata["elements"]["images"]),
                        "total_elements": sum([
                            len(metadata["elements"]["text"]),
                            len(metadata["elements"]["tables"]),
                            len(metadata["elements"]["images"])
                        ])
                    })

                    documents.append(Document(
                        page_content=combined_content,
                        metadata=metadata
                    ))

                    print(f"Page {page_number} processed: {metadata['text_count']} text elements, "
                          f"{metadata['table_count']} tables, {metadata['image_count']} images")

            # Cleanup
            doc.close()
            plumber_doc.close()

            print(f"Total documents created: {len(documents)}")
            return documents

        except Exception as e:
            print(f"PDF processing error: {e}")
            return []

# State Definition for LangGraph
class RAGState(TypedDict):
    """State definition for the RAG workflow"""
    query: str
    retrieved_docs: List[Document]
    context: str
    response: str
    image_references: List[str]
    images_displayed: bool
    metadata: Dict[str, Any]
    step: str
    error: Optional[str]

# Advanced Chunking with LangChain
def chunk_documents_langchain(docs):
    """Advanced chunking using LangChain's text splitters with isolated image chunks"""
    modified_docs = []

    for doc in docs:
        content = doc.page_content
        image_blocks = []
        text_segments = []

        start = 0
        while True:
            img_start = content.find("[Image:", start)
            if img_start == -1:
                if start < len(content):
                    remaining_text = content[start:].strip()
                    if remaining_text:
                        text_segments.append(remaining_text)
                break

            if img_start > start:
                text_before = content[start:img_start].strip()
                if text_before:
                    text_segments.append(text_before)

            desc_start = content.find("Description:", img_start)
            if desc_start != -1:
                desc_end = content.find("\n\n", desc_start)
                if desc_end == -1:
                    desc_end = len(content)
                else:
                    temp_end = desc_end + 2
                    while temp_end < len(content) and content[temp_end].isspace():
                        temp_end += 1
                    desc_end = temp_end if temp_end < len(content) else len(content)

                image_block = content[img_start:desc_end].strip()
                if image_block:
                    image_blocks.append(image_block)

                start = desc_end
            else:
                start = img_start + len("[Image:")

        combined_text = "\n\n".join(text_segments)

        if combined_text.strip():
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                is_separator_regex=False,
            )
            text_chunks = text_splitter.split_text(combined_text)

            for chunk in text_chunks:
                if chunk.strip():
                    modified_docs.append(Document(page_content=chunk.strip(), metadata=doc.metadata))

        for img_block in image_blocks:
            if img_block.strip():
                modified_docs.append(Document(page_content=img_block.strip(), metadata=doc.metadata))

    return modified_docs

# Create Vector Store with LangChain
def create_vector_store_langchain(documents, embedding_model):
    """Create embeddings and store in Chroma vector database using LangChain"""
    filtered_docs = filter_complex_metadata(documents)
    vectorstore = Chroma.from_documents(
        documents=filtered_docs,
        embedding=embedding_model,
        collection_name="rag_collection"
    )
    return vectorstore

# Image Manager with LangChain Integration
class ImageManager:
    """Advanced image manager integrated with LangChain workflows"""

    def __init__(self, docs):
        self.image_map = {}
        self._build_image_map(docs)

    def _build_image_map(self, docs):
        """Build comprehensive mapping of image references to file paths"""
        for doc in docs:
            if 'elements' in doc.metadata and 'images' in doc.metadata['elements']:
                for img_info in doc.metadata['elements']['images']:
                    page_num = doc.metadata['page_number']
                    img_idx = img_info['index'] + 1

                    ref_formats = [
                        f"page{page_num}_image{img_idx}",
                        f"page{page_num}_img{img_idx}",
                        f"Image_{img_idx}_Page_{page_num}",
                        img_info['name']
                    ]

                    for ref in ref_formats:
                        self.image_map[ref.lower()] = {
                            'path': img_info['saved_path'],
                            'metadata': img_info
                        }

    def get_image_info(self, reference):
        """Get complete image information from reference"""
        return self.image_map.get(reference.lower())

    def get_image_path(self, reference):
        """Get image path from reference"""
        info = self.get_image_info(reference)
        return info['path'] if info else None

    def display_image_chain(self, references: List[str]) -> bool:
        """Display multiple images as part of a chain"""
        displayed_any = False
        for ref in references:
            if self.display_image(ref):
                displayed_any = True
        return displayed_any

    def display_image(self, reference, width=300, height=200):
        """Display image in notebook with enhanced metadata"""
        info = self.get_image_info(reference)
        if info and os.path.exists(info['path']):
            try:
                img = Image.open(info['path'])
                plt.figure(figsize=(width/100, height/100))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Image: {reference}\nPage: {info['metadata'].get('name', 'Unknown')}")
                plt.tight_layout()
                plt.show()
                return True
            except Exception as e:
                print(f"❌ Error displaying image {reference}: {e}")
                return False
        else:
            print(f"❌ Image not found: {reference}")
            return False

# LangChain Prompt Templates
query_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert query analyzer for a RAG system.
    Analyze the user's query and determine:
    1. What type of information they're looking for
    2. Whether they might need visual elements (images, charts, diagrams)
    3. Key terms that should be used for retrieval

    Provide analysis in this format:
    Query Type: [factual/visual/conceptual/procedural]
    Needs Images: [yes/no]
    Key Terms: [comma-separated terms]
    Search Strategy: [broad/specific/multi-aspect]
    """),
    ("human", "Query: {query}")
])

context_synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at synthesizing information from multiple document chunks.
    Your task is to combine the retrieved documents into a coherent context summary.

    Guidelines:
    1. Maintain important details and relationships
    2. Preserve image references exactly as they appear
    3. Organize information logically
    4. Remove redundancy while keeping completeness
    """),
    ("human", """Retrieved Documents:
    {documents}

    User Query: {query}

    Synthesized Context:""")
])

response_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant providing comprehensive answers based on document context.

    Instructions:
    1. Answer the question thoroughly using the provided context
    2. When you reference visual elements, wrap them in special tags like [IMAGE:page1_img1] so they can be rendered
    3. If context contains image references like [Image: Image_1_Page_1], convert them to [IMAGE:page1_img1] format
    4. Be specific and cite relevant parts of the context
    5. If the context is insufficient, acknowledge limitations

    Context: {context}
    """),
    ("human", "Question: {query}")
])

print("✅ LangChain prompt templates created")

class StreamlitRAGChain:
    """LangChain-based RAG optimized for Streamlit - Fixed version without Streamlit context issues"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.vectorstore = None
        self.chain = None
        self.image_manager = None
        # Ensure memory knows which output to save when the chain returns multiple keys
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
        """Initialize the complete RAG system - Returns success status and error message"""
        try:
            # Debug environment variables first
            env_ok, missing_vars = debug_environment_variables()
            if not env_ok:
                error_msg = f"Missing environment variables: {', '.join(missing_vars)}"
                print(f"❌ {error_msg}")
                self.initialization_error = error_msg
                return False, error_msg
            
            # Force reload environment variables
            load_dotenv(override=True)
            
            print("🔄 Loading PDF documents...")
            loader = HybridPDFLoader(self.pdf_path)
            docs = loader.load()
            print(f"✅ Loaded {len(docs)} documents.")

            print("🔄 Chunking documents...")
            modified_docs = chunk_documents_langchain(docs)
            print(f"✅ Chunked into {len(modified_docs)} chunks.")

            print("🔄 Creating embeddings...")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="intfloat/e5-large-v2",
                encode_kwargs={"normalize_embeddings": True},
            )
            
            print("🔄 Creating vector store...")
            self.vectorstore = create_vector_store_langchain(modified_docs, self.embedding_model)
            print("✅ Vector store created.")

            print("🔄 Initializing image manager...")
            self.image_manager = ImageManager(docs)
            print("✅ Image manager initialized.")

            print("🔄 Connecting to Azure OpenAI...")
            # Check each variable individually
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            
            self.llm = AzureChatOpenAI(
                deployment_name=deployment,
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
                temperature=0
            )
            
            # Test the connection
            test_response = self.llm.invoke([HumanMessage(content="Hello")])
            print(f"✅ Connection test successful: {test_response.content[:50]}...")
            
            print("🔄 Creating RAG chain...")
            # Create a custom prompt that preserves image references
            from langchain.prompts import PromptTemplate

            # Custom prompt for question answering that preserves image references  
            qa_prompt = PromptTemplate(
                template="""Use the following pieces of context to answer the question. 
                When you see image references like [Image: Image_1_Page_1], convert them to [IMAGE:page1_img1] format for rendering.
                
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
            print("✅ RAG system initialized successfully!")
            return True, "Success"
            
        except Exception as e:
            error_msg = f"Initialization failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.initialization_error = error_msg
            return False, error_msg

    def query(self, user_input: str) -> Dict[str, Any]:
        """Process query and return results with better error handling"""
        if not self.initialized:
            return {
                "response": f"RAG system not initialized. {self.initialization_error or 'Please check initialization.'}",
                "images": [],
                "sources": []
            }

        try:
            print(f"Processing query: {user_input}")

            # Build chat history explicitly to avoid multiple output key error
            chat_history = []
            try:
                if hasattr(self.memory, "chat_memory") and getattr(self.memory.chat_memory, "messages", None):
                    chat_history = [m.content for m in self.memory.chat_memory.messages if getattr(m, "content", None) is not None]
            except Exception:
                chat_history = []

            # Use the chain with proper input format
            result = self.chain({
                "question": user_input,
                "chat_history": chat_history
            })


            print(f"Chain result keys: {result.keys()}")

            
            response_text = result.get("answer", "No answer found.")
            source_documents = result.get("source_documents", [])

            print(f"Response: {response_text[:100]}...")
            print(f"Found {len(source_documents)} source documents")

            # Extract image references from the response text and replace with actual images
            img_refs_raw = re.findall(r"\[IMAGE:\s*([^\]\n]+)\]", response_text)

            # Also look for plain image references like "page1_image1" in the text
            plain_img_refs = re.findall(r"\b(page\d+_img\d+|page\d+_image\d+|Image_\d+_Page_\d+)\b", response_text)

            all_refs = img_refs_raw + plain_img_refs

            image_references = []
            for ref in all_refs:
                ref_clean = ref.strip()
                info = None
                if self.image_manager:
                    # Try multiple lookup patterns
                    info = self.image_manager.get_image_info(ref_clean) \
                        or self.image_manager.get_image_info(ref_clean.lower()) \
                        or self.image_manager.get_image_info(ref_clean.replace('_image', '_img')) \
                        or self.image_manager.get_image_info(ref_clean.replace('_img', '_image'))

                if info and info.get("path") and os.path.exists(info["path"]):
                    saved_path = info["path"]
                    meta = dict(info.get("metadata") or {})
                    
                    image_references.append({
                        "ref": ref_clean,
                        "path": saved_path,
                        "metadata": meta,
                        "exists": True
                    })
                    
                    # Replace the text reference with an image placeholder in the response
                    response_text = response_text.replace(ref_clean, f"[IMAGE:{ref_clean}]")
                    response_text = response_text.replace(f"[IMAGE:{ref_clean}]", f"[IMAGE:{ref_clean}]")
                else:
                    image_references.append({
                        "ref": ref_clean,
                        "path": None,
                        "metadata": None,
                        "exists": False
                    })

            # Extract sources from source documents (unchanged logic)
            sources = []
            for doc in source_documents:
                source_info = f"Page {doc.metadata.get('page_number', 'N/A')}"
                if doc.metadata.get('total_pages'):
                    source_info += f" of {doc.metadata.get('total_pages')}"
                if doc.metadata.get('source'):
                    source_info += f" from {os.path.basename(doc.metadata.get('source'))}"
                sources.append(source_info)

            # Log what we found (don't try to render images here — leave rendering to frontend)
            if image_references:
                print(f"Image references parsed: {[ir['ref'] for ir in image_references]}")
            else:
                print("No image references found in the response.")


            return {
                "response": response_text,
                "images": image_references,
                "sources": list(set(sources))
                }

        except Exception as e:
            print(f"Error during query processing: {e}")
            import traceback
            traceback.print_exc()
            return {"response": f"An error occurred: {e}", "images": [], "sources": []}

    def get_chat_history(self) -> List[BaseMessage]:
        """Get conversation history"""
        return self.memory.chat_memory.messages

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        print("Conversation memory cleared.")

# Note: Remove the @st.cache_resource decorator since streamlit is not imported here
def get_rag_engine(pdf_path: str):
    """RAG engine initialization function"""
    engine = StreamlitRAGChain(pdf_path)
    success, error_msg = engine.initialize_rag()
    if success:
        return engine
    else:
        print(f"Failed to initialize RAG engine: {error_msg}")
        return None