---
title: AI Insurance Voice Agent
emoji: 🦜
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
python_version: "3.11"
---

# 🦜 AI Insurance Voice Agent

A LangGraph-based voice-to-voice agent by Team Outliers (IIT BHU).

## Features
- Upload any company's PDF policies or URLs as the knowledge base
- Query via voice or text — responses returned in both text and audio
- Smart routing: Vector DB → Google Search → LLM fallback
- Policy comparison across multiple uploaded documents

## Stack
| Component | Tool |
|-----------|------|
| STT | Groq Whisper API (whisper-large-v3-turbo) |
| LLM | LLaMA 3.3 70B via Groq |
| Router | LLaMA 3.1 8B via Groq |
| Vector DB | AstraDB (Cassandra) |
| Embeddings | all-MiniLM-L6-v2 |
| TTS | gTTS |
| Framework | LangGraph + LangChain |
| UI | Gradio |