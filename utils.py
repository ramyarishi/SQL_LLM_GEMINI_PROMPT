from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def build_rag_chain(text):
    # Split text into chunks
    splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Create embeddings using HuggingFace sentence transformer
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS vectorstore
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # Setup HuggingFace text-generation pipeline with Flan-T5
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0.7,
        do_sample=True,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Build RetrievalQA chain with retriever
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
    )

    return chain
import PyPDF2

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
    return raw_text

