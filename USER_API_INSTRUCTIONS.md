# User's Instruction: Using the API Only

This document describes how to use **Image Reader**, **PDF Reader**, **Web Crawler**, **Gathering**, **Conversations**, **Customization**, **Agents (with RAG)**, and **Flows** by calling the APIs directly (no UI). All examples use `http://localhost:8000` as the base URL. Replace with your server URL if different.

**Interactive API docs (Swagger):** Open `http://localhost:8000/docs` in a browser to see full endpoint descriptions, request/response schemas, and try requests from the UI.

---

## Table of Contents

1. [Image Reader](#1-image-reader)
2. [PDF Reader](#2-pdf-reader)
3. [Web Crawler](#3-web-crawler)
4. [Gathering](#4-gathering)
5. [Conversations](#5-conversations)
6. [Customization](#6-customization)
7. [Agents with RAG](#7-agents-with-rag)
8. [Flows](#8-flows)

---

## 1. Image Reader

Image Reader uses Qwen Vision for OCR and optional AI processing (Gemini, Qwen, Mistral). Endpoints are under `/image-reader/`.

### 1.1 Get available providers and models

Use these before calling read-and-process so you know valid `provider` and `model` values.

```bash
# List LLM providers (e.g. gemini, qwen, mistral)
curl -s http://localhost:8000/providers | jq

# List available models (per provider)
curl -s http://localhost:8000/models | jq
```

### 1.2 Read text from a single image (OCR only)

Extract text from one image. No AI processing.

```bash
curl -X POST http://localhost:8000/image-reader/read \
  -F "file=@/path/to/your/image.jpg"
```

With optional custom OCR prompt:

```bash
curl -X POST http://localhost:8000/image-reader/read \
  -F "file=@/path/to/your/image.png" \
  -F "prompt=Extract only the handwritten text"
```

**Response (success):** `{"success": true, "text": "...", "image_info": {...}, "timestamp": "..."}`

### 1.3 Read text from multiple images (OCR only)

Extract text from up to 5 images. Each image’s text is returned separately.

```bash
curl -X POST http://localhost:8000/image-reader/read-multiple \
  -F "files=@/path/to/image1.jpg" \
  -F "files=@/path/to/image2.png"
```

**Response:** `{"success": true, "total_images": 2, "results": [{"text": "...", "image_index": 1}, ...], "timestamp": "..."}`

### 1.4 Read one image and process with AI

OCR one image, then send the extracted text to an LLM with your system prompt.

```bash
curl -X POST http://localhost:8000/image-reader/read-and-process \
  -F "file=@/path/to/your/image.jpg" \
  -F "system_prompt=Summarize the following content in 3 bullet points." \
  -F "provider=qwen" \
  -F "model=qwen-vl-plus"
```

Optional OCR prompt:

```bash
curl -X POST http://localhost:8000/image-reader/read-and-process \
  -F "file=@/path/to/your/image.jpg" \
  -F "system_prompt=Summarize the key points." \
  -F "provider=gemini" \
  -F "model=gemini-1.5-flash" \
  -F "ocr_prompt=Extract all text from this screenshot"
```

**Response:** `{"success": true, "extracted_text": "...", "ai_result": "...", "provider": "qwen", "model": "qwen-vl-plus", "system_prompt": "...", "timestamp": "..."}`

### 1.5 Read multiple images and process with AI (one combined result)

OCR up to 5 images, concatenate their text, then run AI once on the combined text.

```bash
curl -X POST http://localhost:8000/image-reader/read-and-process-multiple \
  -F "files=@/path/to/image1.jpg" \
  -F "files=@/path/to/image2.png" \
  -F "system_prompt=List the main topics covered across all content." \
  -F "provider=qwen" \
  -F "model=qwen-vl-plus"
```

**Response:** Same shape as single-image read-and-process; `extracted_text` is the combined text, and may include `image_count`.

---

## 2. PDF Reader

PDF Reader extracts text from a PDF and processes it with an LLM using your system prompt.

### 2.1 Read and process a PDF

**Required:** PDF file and `system_prompt`. Optional: `llm_provider`, `model_name` (defaults from server if omitted).

```bash
curl -X POST http://localhost:8000/pdf-reader/read \
  -F "file=@/path/to/document.pdf" \
  -F "system_prompt=Summarize the key points of this document in 5 bullet points."
```

With explicit provider and model:

```bash
curl -X POST http://localhost:8000/pdf-reader/read \
  -F "file=@/path/to/document.pdf" \
  -F "system_prompt=Extract all section headings and write a short abstract." \
  -F "llm_provider=gemini" \
  -F "model_name=gemini-1.5-flash"
```

**Response (success):** `{"success": true, "extracted_text": "...", "extracted_text_length": 5000, "page_count": 10, "ai_result": "...", "provider": "gemini", "model": "...", "system_prompt": "...", "timestamp": "..."}`

---

## 3. Web Crawler

The crawler fetches a website, extracts content, uses AI to clean and structure it, and saves the result to a RAG collection.

### 3.1 Crawl a URL (one-off)

Crawl a single URL and save to a RAG collection. Collection name and description can be auto-generated if omitted.

```bash
curl -X POST http://localhost:8000/crawler/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "use_js": false,
    "collection_name": "my_example_site",
    "collection_description": "Content from example.com"
  }'
```

With more options (follow links, limit pages):

```bash
curl -X POST http://localhost:8000/crawler/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/docs",
    "use_js": false,
    "follow_links": true,
    "max_depth": 2,
    "max_pages": 20,
    "same_domain_only": true,
    "collection_name": "example_docs",
    "collection_description": "Documentation from example.com"
  }'
```

**Response:** `{"success": true, "url": "...", "collection_name": "...", "collection_description": "...", "raw_file": "...", "extracted_file": "...", "extracted_data": {...}, "pages_crawled": 5, "total_links_found": 30}`

### 3.2 Crawler profiles (save and reuse)

Create a profile once, then execute it to crawl with the same settings.

**List profiles:**

```bash
curl -s http://localhost:8000/crawler/profiles | jq
```

**Create a profile:**

```bash
curl -X POST http://localhost:8000/crawler/profiles \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Blog Crawler",
    "description": "Crawl my blog for RAG",
    "url": "https://myblog.com",
    "use_js": false,
    "collection_name": "my_blog",
    "collection_description": "Blog posts",
    "follow_links": true,
    "max_depth": 2,
    "max_pages": 50,
    "same_domain_only": true
  }'
```

**Response:** Returns the created profile object including `id` (profile_id).

**Get one profile:**

```bash
curl -s http://localhost:8000/crawler/profiles/YOUR_PROFILE_ID | jq
```

**Update a profile:**

```bash
curl -X PUT http://localhost:8000/crawler/profiles/YOUR_PROFILE_ID \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Blog Crawler Updated",
    "url": "https://myblog.com",
    "max_pages": 100
  }'
```

**Execute a profile (crawl using saved config):**

```bash
curl -X POST http://localhost:8000/crawler/profiles/YOUR_PROFILE_ID/execute
```

**Delete a profile:**

```bash
curl -X DELETE http://localhost:8000/crawler/profiles/YOUR_PROFILE_ID
```

---

## 4. Gathering

Gathering uses AI to research a topic by querying **Wikipedia**, **Reddit** (via web search with `site:reddit.com`), and **general web search**. The AI synthesizes results into a markdown report. Limits (e.g. `max_iterations`) prevent infinite searching.

### 4.1 Gather data

```bash
curl -X POST http://localhost:8000/gathering/gather \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Best practices for learning Python in 2024",
    "max_iterations": 10
  }'
```

With optional provider and model:

```bash
curl -X POST http://localhost:8000/gathering/gather \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the pros and cons of electric vehicles?",
    "max_iterations": 12,
    "llm_provider": "qwen",
    "model_name": "qwen-plus",
    "max_tokens": 8192,
    "temperature": 0.5
  }'
```

**Response:** `{"success": true, "content": "## Summary\n\n...", "provider": "qwen", "model": "qwen-plus", "max_iterations": 10}`

**Limits:**
- `max_iterations`: 3–20 (default: 10). Stops the AI from searching forever.
- `max_tokens`: 512–32768 (default: 8192)
- `temperature`: 0.0–1.0 (default: 0.5)

---

## 5. Conversations

Conversations let two AI models talk to each other (and optionally the user) for a fixed number of turns. You create a **configuration** (two models + system prompts), then **start** a session with a topic and **continue** until max turns.

### 5.1 Create a conversation configuration

Create a config that defines model 1, model 2, and max turns. You need this `config_id` to start a session.

```bash
curl -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Science Debate",
    "description": "Two models debate a science topic",
    "config": {
      "model1_config": {
        "provider": "mistral",
        "model_name": "mistral-large-latest",
        "system_prompt": "You are a physicist. Argue from a physics perspective. Be concise."
      },
      "model2_config": {
        "provider": "qwen",
        "model_name": "qwen-plus",
        "system_prompt": "You are a biologist. Argue from a biology perspective. Be concise."
      },
      "max_turns": 10
    }
  }'
```

**Response:** `{"id": "CONFIG_ID_HERE", "message": "Conversation configuration created successfully"}`. Save `id` as `config_id`.

### 5.2 List and get configurations

```bash
# List all configurations
curl -s http://localhost:8000/conversations | jq

# Get one configuration by config_id
curl -s http://localhost:8000/conversations/CONFIG_ID | jq
```

### 5.3 Start a conversation session

Start a new session with a topic. The two models will begin alternating.

```bash
curl -X POST http://localhost:8000/conversations/start \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "CONFIG_ID",
    "topic": "Is consciousness purely physical?"
  }'
```

**Response:** Includes `session_id`, `turn_number`, `max_turns`, `is_complete`, `messages`, `conversation_history`. Save `session_id` for continue and history.

### 5.4 Continue a conversation

Send another turn. You can optionally inject a `user_message`; otherwise the models continue between themselves.

```bash
curl -X POST http://localhost:8000/conversations/continue \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "SESSION_ID",
    "user_message": "What do you both think about emergence?"
  }'
```

Without user message (models only):

```bash
curl -X POST http://localhost:8000/conversations/continue \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID"}'
```

**Response:** Same shape as start; `turn_number` and `conversation_history` update each time.

### 5.5 Get conversation history

Retrieve the full history for a session.

```bash
curl -s http://localhost:8000/conversations/history/SESSION_ID | jq
```

**Response:** `{"session_id": "...", "config_id": "...", "config_name": "...", "started_at": "...", "total_turns": 5, "conversation_history": [...]}`

### 5.6 List saved conversations and get content

If the server saves conversation transcripts to files:

```bash
# List saved conversation files
curl -s http://localhost:8000/conversations/saved | jq

# Get content of a saved file (filename from list above)
curl -s "http://localhost:8000/conversations/saved/FILENAME.txt" | jq
```

### 5.7 Update and delete a configuration

```bash
# Update (same body shape as create, partial ok depending on implementation)
curl -X PUT http://localhost:8000/conversations/CONFIG_ID \
  -H "Content-Type: application/json" \
  -d '{"name": "New Name", "config": { ... }}'

# Delete
curl -X DELETE http://localhost:8000/conversations/CONFIG_ID
```

---

## 6. Customization

Customization profiles let you define reusable AI behavior: a **system prompt** plus an optional **RAG collection** for context. You create a profile once, then **query** it with short user prompts; the AI uses the profile’s system prompt and (if set) RAG context to answer.

### 6.1 Create a customization profile

```bash
curl -X POST http://localhost:8000/customizations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "FAQ Assistant",
    "description": "Answers from our FAQ knowledge base",
    "system_prompt": "You are a helpful support assistant. Answer using only the provided context. If the answer is not in the context, say so.",
    "rag_collection": "my_faq_collection",
    "llm_provider": "qwen",
    "model_name": "qwen-plus"
  }'
```

**Response:** Returns the created profile (includes `id`). Save the `id` for querying.

- **rag_collection**: Optional. RAG collection name; if set, the query will pull context from it.
- **llm_provider** / **model_name**: Optional overrides; omit to use server defaults.

### 6.2 List and get customizations

```bash
# List all customization profiles
curl -s http://localhost:8000/customizations | jq

# Get one profile by ID
curl -s http://localhost:8000/customizations/PROFILE_ID | jq
```

### 6.3 Query a customization (with RAG)

Send a user query; the API uses the profile’s system prompt and optional RAG context.

```bash
curl -X POST http://localhost:8000/customizations/PROFILE_ID/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your return policy?",
    "n_results": 5
  }'
```

**Response:** `{"response": "...", "profile_id": "...", "profile_name": "...", "model_used": "...", "rag_collection_used": "my_faq_collection", "metadata": {...}}`

- **n_results**: Number of RAG documents to use (1–20, default 3).

### 6.4 Update and delete

```bash
# Update (same body shape as create)
curl -X PUT http://localhost:8000/customizations/PROFILE_ID \
  -H "Content-Type: application/json" \
  -d '{"name": "FAQ Assistant v2", "system_prompt": "...", "rag_collection": "my_faq_collection"}'

# Delete
curl -X DELETE http://localhost:8000/customizations/PROFILE_ID
```

---

## 7. Agents with RAG

Agents are AI assistants that can use **RAG collections** (for knowledge) and **tools** (e.g. web search, calculator). You create an agent with `rag_collections` and `tools`, then **run** it with a query.

### 7.1 Create an agent (with RAG and tools)

```bash
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research Agent",
    "description": "Agent with RAG and web search",
    "agent_type": "hybrid",
    "llm_provider": "qwen",
    "model_name": "qwen-plus",
    "rag_collections": ["my_docs", "my_faq_collection"],
    "tools": ["web_search", "wikipedia"],
    "system_prompt": "You are a research assistant. Use the provided context and tools to answer. Cite sources when possible.",
    "temperature": 0.5,
    "max_tokens": 8192
  }'
```

**Response:** Returns created agent (includes `id`). Use this `agent_id` for run/update/delete.

- **agent_type**: `"rag"`, `"tool"`, or `"hybrid"` (RAG + tools).
- **rag_collections**: List of RAG collection names to use as context.
- **tools**: List of tool IDs, e.g. `web_search`, `wikipedia`, `calculator`, `crawler`.
- **system_prompt**: Optional; defines agent behavior.

### 7.2 List and get agents

```bash
# List all agents
curl -s http://localhost:8000/agents | jq

# Get one agent by ID
curl -s http://localhost:8000/agents/AGENT_ID | jq
```

### 7.3 Run an agent

```bash
curl -X POST http://localhost:8000/agents/AGENT_ID/run \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the key points from our docs about onboarding"}'
```

**Response:** `{"response": "...", "sources": [...], "metadata": {...}}`

Optional **context** for the run:

```bash
curl -X POST http://localhost:8000/agents/AGENT_ID/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What did we decide about feature X?", "context": {"thread_id": "123"}}'
```

### 7.4 Run agent with streaming

```bash
curl -X POST http://localhost:8000/agents/AGENT_ID/run/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain our refund policy"}' \
  -H "Accept: text/plain"
```

Returns streaming text (e.g. `text/plain`).

### 7.5 Update and delete agent

```bash
# Update (same body shape as create)
curl -X PUT http://localhost:8000/agents/AGENT_ID \
  -H "Content-Type: application/json" \
  -d '{"name": "Research Agent v2", "rag_collections": ["my_docs"], "tools": ["web_search"]}'

# Delete
curl -X DELETE http://localhost:8000/agents/AGENT_ID
```

---

## 8. Flows

Flows chain steps together: **Customization**, **Agent**, **DB Tool**, **Request**, **Crawler**, or **Dialogue**. Each step can use the previous step’s output as input. Create a flow with an ordered list of steps, then **execute** it with optional initial input.

### 8.1 List and get flows

```bash
# List all flows
curl -s http://localhost:8000/flows | jq

# Get one flow by ID
curl -s http://localhost:8000/flows/FLOW_ID | jq
```

### 8.2 Create a flow

Each step has: **step_id**, **step_type**, **step_name**, **resource_id** (ID of the customization, agent, db_tool, request, or crawler), and optionally **use_previous_output** (use previous step’s output as input).

**Step types:** `customization`, `agent`, `db_tool`, `request`, `crawler`, `dialogue`.

```bash
curl -X POST http://localhost:8000/flows \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research then Summarize",
    "description": "Query RAG via customization then summarize with an agent",
    "steps": [
      {
        "step_id": "step1",
        "step_type": "customization",
        "step_name": "Get RAG context",
        "resource_id": "PROFILE_ID",
        "input_query": "Key points about product X",
        "use_previous_output": false
      },
      {
        "step_id": "step2",
        "step_type": "agent",
        "step_name": "Summarize",
        "resource_id": "AGENT_ID",
        "use_previous_output": true
      }
    ],
    "is_active": true
  }'
```

**Response:** `{"flow_id": "...", "message": "Flow created successfully"}`

- **resource_id**: For `customization` use customization profile ID; for `agent` use agent ID; for `db_tool` use db-tool ID; for `request` use request-tool ID; for `crawler` use crawler profile ID; for `dialogue` use dialogue ID.
- **use_previous_output**: If `true`, this step receives the previous step’s output as input.

### 8.3 Execute a flow

```bash
curl -X POST http://localhost:8000/flows/FLOW_ID/execute \
  -H "Content-Type: application/json" \
  -d '{
    "initial_input": "Optional initial prompt for the first step",
    "context": {}
  }'
```

**Response:** `{"flow_id": "...", "flow_name": "...", "success": true, "step_results": [...], "final_output": "...", "total_execution_time": 1.23}`

- **initial_input**: Optional; used as input for the first step if the step expects it.
- **context**: Optional key-value context for execution.

### 8.4 Update and delete flow

```bash
# Update (same body shape as create)
curl -X PUT http://localhost:8000/flows/FLOW_ID \
  -H "Content-Type: application/json" \
  -d '{"name": "New name", "steps": [...]}'

# Delete
curl -X DELETE http://localhost:8000/flows/FLOW_ID
```

---

## Quick reference

| Area            | Base path              | Main actions |
|-----------------|------------------------|-------------|
| Image Reader    | `/image-reader/`       | `POST read`, `POST read-multiple`, `POST read-and-process`, `POST read-and-process-multiple` |
| PDF Reader      | `/pdf-reader/`         | `POST read` (file + system_prompt) |
| Web Crawler     | `/crawler/`            | `POST crawl`, `GET/POST/PUT/DELETE profiles`, `POST profiles/{id}/execute` |
| Gathering       | `/gathering/`          | `POST gather` (prompt, max_iterations, optional provider/model) |
| Conversations   | `/conversations/`      | `POST` create, `GET` list, `GET {id}`, `POST start`, `POST continue`, `GET history/{session_id}` |
| Customization   | `/customizations/`    | `POST` create, `GET` list, `GET {id}`, `PUT` update, `DELETE` delete, `POST {id}/query` (with RAG) |
| Agents          | `/agents/`             | `POST` create (rag_collections, tools), `GET` list, `GET {id}`, `PUT` update, `DELETE` delete, `POST {id}/run`, `POST {id}/run/stream` |
| Flows           | `/flows/`              | `GET` list, `GET {id}`, `POST` create (steps), `PUT` update, `DELETE` delete, `POST {id}/execute` |

For full request/response schemas and try-it-out, use **Swagger UI**: `http://localhost:8000/docs`.
