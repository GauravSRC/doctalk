import os
import random
from tempfile import NamedTemporaryFile
from typing import List, Literal
from typing_extensions import TypedDict

# Load .env locally — on HF Spaces, secrets are already in the environment
from dotenv import load_dotenv
load_dotenv()

# Must be set before any langchain import that uses WebBaseLoader
os.environ["USER_AGENT"] = "InsuranceAgent/1.0"

# ── Read all secrets from environment ────────────────────────────────────────
GROQ_API_KEY               = os.getenv("GROQ_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID                = os.getenv("ASTRA_DB_ID")

os.environ["TAVILY_API_KEY"]        = os.getenv("TAVILY_API_KEY", "")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"]  = "true"
os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]     = "insurance_agent_v2"

# ── Imports ───────────────────────────────────────────────────────────────────
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Cassandra
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import cassio
from groq import Groq
from gtts import gTTS
import pdfplumber
import gradio as gr

# ── Validate required secrets are present ────────────────────────────────────
missing = [k for k in ["GROQ_API_KEY", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_ID"]
           if not os.getenv(k)]
if missing:
    raise EnvironmentError(f"Missing required secrets: {missing}")

# ── CassIO / AstraDB init ────────────────────────────────────────────────────
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
print("✅ CassIO initialised")

# ── Groq client (used for both STT and LLM) ──────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

# ── STT: Groq Whisper API (replaces local Whisper — works on CPU) ────────────
def speech_to_text(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        transcription = groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
        )
    return transcription.text

# ── TTS: gTTS ────────────────────────────────────────────────────────────────
def text_to_speech(text: str) -> str:
    tts = gTTS(text)
    out = NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(out.name)
    return out.name

# ── Embeddings ────────────────────────────────────────────────────────────────
import time

# AFTER (replace with this):
os.environ.pop("HF_ENDPOINT", None)               # remove mirror — it's blocked
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"     # 10 minutes for slow connections

def _load_embeddings(retries: int = 3, delay: int = 10):
    """Load embeddings with retry logic to handle network timeouts gracefully."""
    for attempt in range(1, retries + 1):
        try:
            print(f"Loading embedding model (attempt {attempt}/{retries})...")
            emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("✅ Embeddings loaded")
            return emb
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt} failed: {type(e).__name__}")
            if attempt < retries:
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
    raise RuntimeError(
        "Failed to load embeddings after all attempts.\n"
        "Fix: Run  python download_model.py  first to pre-cache the model."
    )

embeddings = _load_embeddings()

# ── AstraDB Vector Store ──────────────────────────────────────────────────────
astra_vdb = Cassandra(
    embedding=embeddings,
    table_name="insurance_agent_v2",
    session=None,
    keyspace=None
)
retriever = astra_vdb.as_retriever(search_kwargs={"k": 3})
print("✅ AstraDB connected (table: insurance_agent_v2)")

# ── LLMs ─────────────────────────────────────────────────────────────────────
router_llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
final_llm  = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
print("✅ LLMs ready")

# ── Tavily Search ─────────────────────────────────────────────────────────────
search = TavilySearchResults(max_results=5)

# ── Text Splitter ─────────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)

# ─────────────────────────────────────────────────────────────────────────────
# INGESTION
# ─────────────────────────────────────────────────────────────────────────────
def ingest_pdfs(pdf_paths: List[str]) -> int:
    chunks = []
    messages = []
    for path in pdf_paths:
        try:
            with pdfplumber.open(path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            source_name = os.path.basename(path)
            doc = Document(
                page_content=text,
                metadata={"source": source_name, "type": "pdf"}
            )
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
            messages.append(f"  ✅ {source_name} → {len(doc_chunks)} chunks")
        except Exception as e:
            messages.append(f"  ❌ {os.path.basename(path)}: {e}")
    if chunks:
        astra_vdb.add_documents(chunks)
    return len(chunks), "\n".join(messages)


def ingest_urls(urls: List[str]) -> int:
    chunks = []
    messages = []
    for url in urls:
        try:
            docs = WebBaseLoader(url).load()
            for doc in docs:
                doc.metadata["source"] = url
                doc.metadata["type"]   = "url"
            url_chunks = text_splitter.split_documents(docs)
            chunks.extend(url_chunks)
            messages.append(f"  ✅ {url} → {len(url_chunks)} chunks")
        except Exception as e:
            messages.append(f"  ❌ {url}: {e}")
    if chunks:
        astra_vdb.add_documents(chunks)
    return len(chunks), "\n".join(messages)


def ingest_documents(pdf_paths: List[str] = None, urls: List[str] = None):
    total   = 0
    log     = []
    if pdf_paths:
        n, msg = ingest_pdfs(pdf_paths)
        total += n
        log.append(f"PDFs:\n{msg}")
    if urls:
        n, msg = ingest_urls(urls)
        total += n
        log.append(f"URLs:\n{msg}")
    return total, "\n\n".join(log)

# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH
# ─────────────────────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    documents:     str
    question:      str
    generation:    str
    is_comparison: bool


class RouteQuery(BaseModel):
    datasource: Literal["vector-database", "google-search", "llm"] = Field(
        description=(
            "Route to 'vector-database' for questions about uploaded documents or policies. "
            "Route to 'google-search' for current events or general web information. "
            "Route to 'llm' for general conversational or instructional queries."
        )
    )


router_structured = router_llm.with_structured_output(RouteQuery)
router_prompt = ChatPromptTemplate([
    ("system",
     "You are an expert at routing user questions to the appropriate data source. "
     "Prefer 'vector-database' for anything about uploaded company documents or policies. "
     "Use 'google-search' for current events or general web lookups. "
     "Use 'llm' only for general conversational queries."),
    ("user", "{query}")
])
Router = router_prompt | router_structured

COMPARISON_KEYWORDS = [
    "compare", "comparison", "vs", "versus",
    "difference between", "better than", "which is better"
]

def _is_comparison(question: str) -> bool:
    return any(kw in question.lower() for kw in COMPARISON_KEYWORDS)


def vectorstore_node(state: GraphState) -> dict:
    question = state["question"]
    docs     = retriever.invoke(question)

    if _is_comparison(question):
        grouped = {}
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            grouped.setdefault(src, []).append(doc.page_content)
        combined = ""
        for src, contents in grouped.items():
            combined += f"\n\n--- Source: {src} ---\n" + " ".join(contents)
        return {"documents": combined, "question": question, "is_comparison": True}

    sources  = list({doc.metadata.get("source", "unknown") for doc in docs})
    combined = f"[Sources: {', '.join(sources)}]\n\n" + \
               " ".join(doc.page_content for doc in docs)
    return {"documents": combined, "question": question, "is_comparison": False}


def google_search_node(state: GraphState) -> dict:
    question = state["question"]
    result   = search.invoke(question)
    combined = "\n".join(item["content"] for item in result)
    return {"documents": combined, "question": question, "is_comparison": False}


def llm_node(state: GraphState) -> dict:
    question = state["question"]
    response = final_llm.invoke(question)
    return {"documents": "", "question": question,
            "generation": response.content, "is_comparison": False}


_base_template = """You are a knowledgeable assistant. Answer the question using the context below.
If the context lacks sufficient information, use your own general knowledge.
Mention the source of information when it is present in the context.

Context:
{context}

Question: {question}

Answer:"""

_comparison_template = """You are a knowledgeable assistant. The user wants a comparison.
Using the context below (organised by source document), provide a clear structured comparison.
Use a table or bullet points where helpful.

Context:
{context}

Question: {question}

Comparison:"""

_base_chain       = ChatPromptTemplate.from_template(_base_template)       | final_llm | StrOutputParser()
_comparison_chain = ChatPromptTemplate.from_template(_comparison_template) | final_llm | StrOutputParser()


def generator_node(state: GraphState) -> dict:
    is_comp = state.get("is_comparison", False)
    chain   = _comparison_chain if is_comp else _base_chain
    output  = chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"generation": output}


def route_query(state: GraphState) -> str:
    source = Router.invoke({"query": state["question"]})
    return {"vector-database": "vectorstore",
            "google-search":   "google_search",
            "llm":             "llm"}[source.datasource]


memory   = MemorySaver()
workflow = StateGraph(GraphState)

workflow.add_node("vectorstore",   vectorstore_node)
workflow.add_node("google_search", google_search_node)
workflow.add_node("llm",           llm_node)
workflow.add_node("generator",     generator_node)

workflow.add_conditional_edges(
    START, route_query,
    {"vectorstore": "vectorstore", "google_search": "google_search", "llm": "llm"}
)
workflow.add_edge("vectorstore",   "generator")
workflow.add_edge("google_search", "generator")
workflow.add_edge("llm",           END)
workflow.add_edge("generator",     END)

app = workflow.compile(checkpointer=memory)
print("✅ LangGraph compiled")

# ─────────────────────────────────────────────────────────────────────────────
# CHATBOT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
_current_thread_id: str = str(random.randint(1, 1_000_000_000))


def new_session() -> str:
    global _current_thread_id
    _current_thread_id = str(random.randint(1, 1_000_000_000))
    return f"✅ New session started (thread: {_current_thread_id})"


def generate_response(user_input: str) -> str:
    events     = app.stream(
        {"question": user_input},
        {"configurable": {"thread_id": _current_thread_id}}
    )
    generation = ""
    for event in events:
        if "generator" in event:
            generation = event["generator"].get("generation", "")
        elif "llm" in event:
            generation = event["llm"].get("generation", "")
    return generation


def chatbot_pipeline(audio_path=None, text_input=None):
    try:
        if audio_path:
            user_text = speech_to_text(audio_path)
        elif text_input and text_input.strip():
            user_text = text_input.strip()
        else:
            return "Please provide an audio file or type a query.", None
        response_text  = generate_response(user_text)
        response_audio = text_to_speech(response_text)
        return response_text, response_audio
    except Exception as e:
        return f"Error: {e}", None

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO HANDLERS
# ─────────────────────────────────────────────────────────────────────────────
def _handle_audio(audio):
    return chatbot_pipeline(audio_path=audio)

def _handle_text(text):
    return chatbot_pipeline(text_input=text)

def _handle_new_session():
    return new_session()

def _handle_ingest(pdf_files, urls_text):
    paths = []
    if pdf_files:
        for f in pdf_files:
            paths.append(f if isinstance(f, str) else f.name)
    urls  = [u.strip() for u in (urls_text or "").split("\n") if u.strip()]
    total, log = ingest_documents(pdf_paths=paths or None, urls=urls or None)
    return f"✅ Ingested {total} total chunks into AstraDB.\n\n{log}"

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="AI Voice Agent") as demo:

    gr.Markdown("# 🦜 AI Voice Agent\nUpload documents first, then query via voice or text.")

    with gr.Tab("📄 Ingest Documents"):
        gr.Markdown("Upload PDFs and/or paste URLs to build the knowledge base.")
        pdf_upload = gr.File(label="PDF Files", file_count="multiple",
                             file_types=[".pdf"])
        url_box    = gr.Textbox(label="URLs (one per line)", lines=4,
                                placeholder="https://example.com/policy-page")
        ingest_btn = gr.Button("Ingest Documents", variant="primary")
        ingest_out = gr.Textbox(label="Status", interactive=False, lines=6)
        ingest_btn.click(_handle_ingest,
                         inputs=[pdf_upload, url_box],
                         outputs=[ingest_out],
                         api_name=False)

    with gr.Tab("🎙️ Voice Chat"):
        gr.Markdown("Record your query. The agent replies in text and audio.")
        audio_in    = gr.Audio(type="filepath", label="Your Voice Query")
        voice_btn   = gr.Button("Submit", variant="primary")
        voice_text  = gr.Textbox(label="Response Text", lines=5)
        voice_audio = gr.Audio(label="Response Audio")
        voice_btn.click(_handle_audio,
                        inputs=[audio_in],
                        outputs=[voice_text, voice_audio],
                        api_name=False)

    with gr.Tab("💬 Text Chat"):
        gr.Markdown("Type your query. The agent replies in text and audio.")
        text_in    = gr.Textbox(label="Your Query", lines=2,
                                placeholder="Compare term plans from HDFC and Axis Bank...")
        text_btn   = gr.Button("Submit", variant="primary")
        text_out   = gr.Textbox(label="Response Text", lines=5)
        text_audio = gr.Audio(label="Response Audio")
        text_btn.click(_handle_text,
                       inputs=[text_in],
                       outputs=[text_out, text_audio],
                       api_name=False)

    with gr.Row():
        session_btn    = gr.Button("🔄 New Conversation Session")
        session_status = gr.Textbox(label="Session", interactive=False,
                                    value=f"Thread: {_current_thread_id}")
        session_btn.click(_handle_new_session,
                          outputs=[session_status],
                          api_name=False)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,        # disables SSR which triggers the api/info calls
    )