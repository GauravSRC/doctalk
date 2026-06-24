# 🎙️ DocTalk -- AI Voice Insurance Agent

A voice-to-voice AI agent built with LangGraph that lets you upload any company's policy documents and query them via voice or text.
🔗 **Live Demo:** https://gsrc-doctalk.hf.space
📁 **Hugging Face Space:** https://huggingface.co/spaces/gsrc/doctalk

---

# What it Does

* Upload **any company's PDF policies or URLs** as the knowledge base (not locked to one company).
* Ask questions via **voice or text**.
* Receive responses in both **text and audio**.
* Automatically **compare policies** across multiple uploaded documents.
* Smart query routing sends requests to the appropriate source.

### Pipeline

```text
Voice Input
      │
      ▼
Speech-to-Text (Whisper)
      │
      ▼
LLM Router
      │
      ├──► Vector Database
      ├──► Google Search
      └──► Direct LLM
               │
               ▼
          Response Generator
               │
               ▼
        Text-to-Speech
               │
               ▼
          Audio Response
```

---

# Tech Stack

| Component       | Tool                                        |
| --------------- | ------------------------------------------- |
| STT             | Groq Whisper API (`whisper-large-v3-turbo`) |
| Router LLM      | LLaMA 3.1 8B via Groq                       |
| Generator LLM   | LLaMA 3.3 70B via Groq                      |
| Vector Database | AstraDB (Cassandra)                         |
| Embeddings      | `all-MiniLM-L6-v2`                          |
| Web Search      | Tavily                                      |
| Text-to-Speech  | gTTS                                        |
| Orchestration   | LangGraph                                   |
| UI              | Gradio                                      |
| Deployment      | Hugging Face Spaces                         |

---

# Run Locally

```bash
git clone https://github.com/GauravSRC/doctalk.git
cd doctalk

python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_key
ASTRA_DB_APPLICATION_TOKEN=your_token
ASTRA_DB_ID=your_db_id
TAVILY_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
```

Run the application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

---

# Architecture

```text
                    User Voice / Text
                           │
                           ▼
              Speech-to-Text (Groq Whisper)
                           │
                           ▼
                LLM Router (LLaMA 3.1 8B)
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   Vector Database     Google Search      Direct LLM
     (AstraDB)           (Tavily)      (LLaMA 3.3 70B)
        │                  │
        └────────────┬─────┘
                     ▼
        Generator (LLaMA 3.3 70B)
                     │
                     ▼
          Text-to-Speech (gTTS)
                     │
                     ▼
               Audio Response
```

---

# Key Features

* **Domain Agnostic** – Works with documents from any bank, insurer, or company; not limited to a single provider.
* **Policy Comparison** – Compare multiple uploaded policies and receive a structured comparison.
* **Source Citations** – Responses include the document or URL used to generate the answer.
* **Persistent Session Memory** – Conversation context is maintained within a session using LangGraph's `MemorySaver`.
