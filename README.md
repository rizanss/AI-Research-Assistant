# 📚 AI Research Assistant - Chat with Your PDFs!

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Powered-yellow.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20(Qwen)-green.svg)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-purple.svg)

*Tonton Demo Aplikasi di Bawah Ini!*

![Research Assistant Demo](./assets/ai-research-assistant-demo.gif)

## ✨ Project Overview

Selamat datang di "AI Librarian"! Ini adalah aplikasi web cerdas yang dibangun menggunakan arsitektur **Retrieval-Augmented Generation (RAG)** untuk memungkinkan Anda "berdialog" dengan dokumen PDF Anda. Unggah satu atau beberapa makalah penelitian, dan Anda bisa langsung mengajukan pertanyaan kompleks untuk mendapatkan jawaban yang akurat, relevan, dan disampaikan secara natural layaknya berbicara dengan seorang asisten ahli.

Proyek ini mendemonstrasikan pipeline AI/MLOps modern dari awal hingga akhir, mulai dari pemrosesan dokumen, pembuatan *vector database*, hingga *deployment* sebagai aplikasi web interaktif yang memiliki UX modern.

---

## 🚀 Key Features

* **Multi-PDF Processing**: Kemampuan untuk mengunggah dan memproses beberapa dokumen PDF sekaligus menggunakan `PyMuPDF`.
* **Semantic Vector Database**: Dokumen dipecah menjadi *chunks* dan diubah menjadi *embeddings* (menggunakan `Sentence-Transformers`), lalu disimpan dalam **FAISS Vector Store** untuk pencarian makna super cepat.
* **Advanced RAG Pipeline**: Menggunakan **Groq API** dengan model **Qwen-32B** untuk mensintesis jawaban baru, memastikan jawaban tetap faktual namun disampaikan secara conversational.
* **Transparent Thinking Process**: Fitur unik di mana pengguna bisa melihat "alur berpikir" AI di dalam *expander* untuk memahami bagaimana AI sampai pada jawabannya.
* **Interactive & Modern UI**: Antarmuka web yang bersih dan intuitif dibangun dengan **Streamlit**, lengkap dengan *chat history* dan alur kerja yang menyerupai chatbot modern.
* **Secure API Key Management**: Menggunakan `st.secrets` untuk manajemen API key yang aman.

---

## 🏛️ Tech Stack & Architecture

Aplikasi ini menggunakan alur kerja RAG yang modern:

**User Uploads PDF(s) ➔ 1. Parse & Chunk Text ➔ 2. Generate & Store Embeddings in FAISS ➔ User Asks Question ➔ 3. Retrieve Relevant Chunks ➔ 4. Augment Prompt with Context ➔ 5. LLM (Groq/Qwen) ➔ 6. Formatted Answer with Thinking Process**

* **UI & Orchestration**: Streamlit
* **LLM API**: Groq (menggunakan model `qwen-qwq-32b`)
* **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2`
* **Vector Store**: `faiss-cpu`
* **PDF Parser**: `PyMuPDF`
* **Core Language**: Python 3.11+

---

## ⚙️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/rizanss/AI-Research-Assistant.git](https://github.com/rizanss/AI-Research-Assistant.git)
    cd AI-Research-Assistant
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows: .\.venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Groq API Key:**
    * Buat folder `.streamlit` di direktori utama.
    * Di dalamnya, buat file `secrets.toml`.
    * Isi file tersebut dengan API key Groq Anda: `GROQ_API_KEY = "gsk_..."`

5.  **Run the application!**
    ```bash
    streamlit run app.py
    ```
    Buka `http://localhost:8501` di browser Anda.

---

## 📬 Contact
* **Author:** Riza Nursyah
* **GitHub:** [rizanss](https://github.com/rizanss)
* **LinkedIn:** [Riza Nursyah](https://www.linkedin.com/in/riza-nursyah-31a6a7221/)
