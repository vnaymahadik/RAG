import streamlit as st
import os
import tempfile
from google.generativeai import configure, GenerativeModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

st.set_page_config(page_title="PDF RAG System", page_icon="📚", layout="wide")

# App title and description
st.title("📚 PDF RAG System with Gemini")
st.markdown("""
Upload a folder of PDF files and ask questions. This app will search through the documents
and provide answers based on the content using Google's Gemini AI models.
""")

# API Key input
api_key = st.text_input("Enter your Gemini API Key", type="password")

# Model selection dropdown
model_options = {
    "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 1.0 Pro": "gemini-1.0-pro",
}
selected_model = st.selectbox("Select Gemini Model", options=list(model_options.keys()))

# Initialize session state for storing document data
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "file_metadata" not in st.session_state:
    st.session_state.file_metadata = {}

# Function to process PDF files
def process_pdfs(uploaded_files):
    all_texts = []
    file_metadata = {}
    progress_bar = st.progress(0)
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress_text = f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}"
        st.text(progress_text)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name
        
        # Store original filename
        file_metadata[file_path] = uploaded_file.name
        
        # Load and process PDF
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add original filename to metadata
            for doc in documents:
                doc.metadata["original_filename"] = uploaded_file.name
            
            all_texts.extend(documents)
            os.unlink(file_path)  # Remove temp file
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    # Split texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(all_texts)
    
    st.success(f"✅ Successfully processed {len(uploaded_files)} PDFs with {len(chunks)} text chunks.")
    
    return chunks, file_metadata

def setup_rag(chunks, api_key):
    # Configure Gemini
    configure(api_key=api_key)
    
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Create vector store
    with st.spinner("Creating vector database..."):
        vector_store = FAISS.from_documents(chunks, embeddings)
    
    st.success("✅ Vector database created successfully!")
    return vector_store

def query_rag(question, vector_store, api_key, model_name):
    # Configure Gemini
    configure(api_key=api_key)
    
    # Search for relevant documents
    docs = vector_store.similarity_search(question, k=5)
    
    # Format context from retrieved documents
    context_parts = []
    sources_detailed = []
    
    for i, doc in enumerate(docs):
        # Get page number and filename
        page_num = doc.metadata.get('page', 'unknown page')
        filename = doc.metadata.get('original_filename', doc.metadata.get('source', 'Unknown'))
        
        # Format the content with source information
        context_parts.append(f"[Document {i+1} - {filename} (Page {page_num})]\n{doc.page_content}")
        
        # Add to detailed sources
        sources_detailed.append({
            "content": doc.page_content,
            "source": filename,
            "page": page_num
        })
    
    context = "\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""
    You are an assistant that answers questions based on the provided context. 
    If the answer is not in the context, say "I don't have enough information to answer this question."
    Always prioritize information from the context over your general knowledge.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer the question based on the context provided. Be concise and accurate.
    After your answer, include a section called "SOURCES USED" where you explicitly mention which of the provided document sections you used to formulate your answer. 
    Reference them by their document number and page.
    """
    
    # Query Gemini with selected model
    model = GenerativeModel(model_name)
    response = model.generate_content(prompt)
    
    return response.text, sources_detailed

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Process button
if uploaded_files and api_key and st.button("Process Documents"):
    if not api_key:
        st.error("Please enter your Gemini API Key.")
    else:
        with st.spinner("Processing documents..."):
            chunks, file_metadata = process_pdfs(uploaded_files)
            st.session_state.vector_store = setup_rag(chunks, api_key)
            st.session_state.file_metadata = file_metadata
            st.session_state.docs_processed = True

# Query section (only show if documents are processed)
if st.session_state.docs_processed:
    st.header("Ask Questions")
    question = st.text_input("Enter your question about the documents")
    
    if question and st.button("Get Answer"):
        with st.spinner(f"Searching for answer using {selected_model}..."):
            model_id = model_options[selected_model]
            answer, sources_detailed = query_rag(question, st.session_state.vector_store, api_key, model_id)
            
            st.subheader("Answer")
            st.write(answer)
            
            # Display detailed source information in an expandable section
            with st.expander("View Source Documents"):
                for i, source in enumerate(sources_detailed):
                    st.markdown(f"**Source {i+1}: {source['source']} (Page {source['page']})**")
                    st.text_area(f"Content {i+1}", source['content'], height=150)

# Add helpful information in the sidebar
st.sidebar.title("About")
st.sidebar.info("""
This app uses Retrieval-Augmented Generation (RAG) to answer questions based on your PDF documents.
1. Upload your PDF files
2. Enter your Gemini API key
3. Select your preferred Gemini model
4. Process the documents 
5. Ask questions about their content

The app will search through the documents for the most relevant information to answer your questions.
""")

# Display model information
st.sidebar.subheader("Model Information")
st.sidebar.markdown("""
- **Gemini 1.5 Pro**: Most capable model, best for complex reasoning
- **Gemini 1.5 Flash**: Faster and more cost-effective
- **Gemini 1.0 Pro**: Original model, good balance of capabilities
""")