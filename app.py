import streamlit as st
import fitz  # PyMuPDF
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# =======================================================================
#  FUNGSI-FUNGSI INTI
# =======================================================================

# Menggunakan cache resource agar objek berat ini hanya di-load sekali
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

def process_pdfs(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                all_text += page.get_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return text_splitter.create_documents([all_text])

# Menggunakan cache resource, dan mengabaikan hashing untuk _embedding_model
@st.cache_resource
def create_vector_store(_docs, _embedding_model):
    return FAISS.from_documents(_docs, _embedding_model)

def get_llm_response(user_question, context, api_key):
    """Generate jawaban menggunakan Groq API dengan model Qwen."""
    client = ChatGroq(
        groq_api_key=api_key,
        model_name='qwen-qwq-32b'
    )
    prompt_template = """
    Anda adalah asisten riset AI yang ahli.
    Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan KONTEKS yang diberikan.
    Jika kamu tidak tahu jawabannya, katakan saja kamu tidak tahu, jangan mencoba mengarang jawaban.
    Jawab dengan gaya bahasa yang natural, membantu, empatik, dan mudah dipahami dalam Bahasa Indonesia.
    Walaupun konteks dalam Bahasa Inggris, tetap jawab dan berpikir dalam Bahasa Indonesia.

    KONTEKS: {context}
    PERTANYAAN: {question}

    JAWABAN ANDA:
    """
    prompt = prompt_template.format(context=context, question=user_question)
    try:
        response = client.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Maaf, terjadi error saat menghubungi Groq API: {e}"

def parse_response(response_text):
    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip().replace("\n", "<br>")
        answer = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    else:
        thinking = "Proses berpikir tidak terdeteksi di output."
        answer = response_text
    return thinking, answer

# =======================================================================
#  TAMPILAN (INTERFACE) STREAMLIT
# =======================================================================

st.set_page_config(page_title="AI Research Assistant", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stExpander {
    border: none;
    max-width: 75%;
}

/* Target langsung summary element */
details summary {
    display: inline-block;
    color: #f0f0f0;
    padding: 8px;
    border-radius: 30px;
    font-weight: 500;
    margin-bottom: 5px;
    cursor: pointer;
    transition: background-color 0.2s ease;
    width: fit-content;
}

/* Remove underline on <p> inside summary */
details summary p {
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

# --- Inisialisasi Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.title("📚 AI Research Assistant")

# --- UI Logika: Tampilkan Upload atau Chat ---
if st.session_state.vectorstore is None:
    st.header("Unggah Dokumen Anda")
    st.info("Aplikasi ini akan menganalisis isi PDF dan memungkinkan Anda untuk bertanya tentang kontennya.")
    
    uploaded_files = st.file_uploader(
        "Pilih satu atau beberapa file PDF",
        accept_multiple_files=True,
        type="pdf",
        label_visibility="collapsed"
    )

    if st.button("Proses & Mulai Chat", use_container_width=True, type="tertiary"):
        if uploaded_files:
            with st.spinner("Memproses dokumen..."):
                embedding_model = load_embedding_model()
                docs_chunked = process_pdfs(uploaded_files)
                if docs_chunked:
                    st.session_state.vectorstore = create_vector_store(docs_chunked, embedding_model)
                    st.session_state.messages = [{"role": "assistant", "content": "Dokumen Anda sudah siap! Apa yang ingin Anda ketahui?"}]
                    st.rerun()
        else:
            st.warning("Mohon unggah setidaknya satu file PDF.")
else:
    # --- INTERFACE CHAT ---
    st.write("")  # Spacer

    # Custom chat history layout
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "assistant-thinking":
            with st.expander("Tampilkan Alur Berpikir AI"):
                st.markdown(f"""
                <div style="
                    color:#a9a9a9;
                    font-style: italic;
                    padding: 10px;
                    border-left: 2px solid #555;
                    max-width: 600px;
                    width: fit-content;
                    border-radius: 8px;
                ">
                    {content}
                </div>
                """, unsafe_allow_html=True)
            continue

        # Bubble warna dan arah
        if role == "user":
            align = "right"
            bubble_color = "#2c3e50"
            border = "#34495e"
        else:  # assistant
            align = "left"
            bubble_color = "#333333"
            border = "#444"

        # Render bubble (user & assistant)
        st.markdown(f"""
        <div style="
            background-color: {bubble_color};
            color: #E0E0E0;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 75%;
            margin: 5px;
            float: {align};
            clear: both;
            border: 1px solid {border};
        ">
            {content}
        </div>
        """, unsafe_allow_html=True)

    # Terima input dari user
    if prompt := st.chat_input("Tanyakan sesuatu tentang dokumen Anda..."):
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except:
            st.error("GROQ_API_KEY tidak ditemukan. Mohon setup file .streamlit/secrets.toml Anda.")
            st.stop()
        
        # Simpan pesan user    
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("AI sedang menganalisis dan berpikir... 🤔"):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            context_docs = retriever.invoke(prompt)
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            response_text = get_llm_response(prompt, context, groq_api_key)
            thinking, answer = parse_response(response_text)

            # Simpan thinking ke message log
            st.session_state.messages.append({"role": "assistant-thinking", "content": thinking})

            # Langsung tampilkan jawaban ke layar pakai markdown
            st.markdown(f"""
            <div style="
                background-color: #333333;
                color: #E0E0E0;
                padding: 10px 15px;
                border-radius: 10px;
                max-width: 75%;
                margin: 5px;
                float: left;
                clear: both;
                border: 1px solid #444;
            ">
                {answer}
            </div>
            """, unsafe_allow_html=True)

            # Simpan raw answer (tanpa bubble) ke session untuk future rerender
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        st.rerun()
