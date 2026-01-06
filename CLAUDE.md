# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot for querying course materials. Uses FastAPI backend, ChromaDB vector storage, and Anthropic Claude API with tool calling.

## Commands

```bash
# Install dependencies
uv sync

# Run server (from project root)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Debug in PyCharm: open backend/debug_server.py and run with debugger
```

App: http://localhost:8000 | API docs: http://localhost:8000/docs

## Architecture

### High-Level Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND (frontend/)                                                        │
│                                                                             │
│  script.js                                                                  │
│  sendMessage() ──► POST /api/query {query, session_id}                      │
│       ▲                                                                     │
│       └─────────── {answer, sources, session_id} ◄─────────────────────┐    │
└────────────────────────────────────────────────────────────────────────│────┘
                                                                         │
┌────────────────────────────────────────────────────────────────────────│────┐
│ BACKEND (backend/)                                                     │    │
│                                                                        │    │
│  app.py                                                                │    │
│  query_documents() ──► rag_system.query()                              │    │
│                              │                                         │    │
│  rag_system.py               │                                         │    │
│  RAGSystem.query() ──────────┴──► ai_generator.generate_response()     │    │
│                                          │                             │    │
│  ai_generator.py                         ▼                             │    │
│  ┌───────────────────────────────────────────────────────────┐         │    │
│  │ 1st Claude API call (with tools)                          │         │    │
│  │    └─► stop_reason == "tool_use"?                         │         │    │
│  │           │                                               │         │    │
│  │           ▼ Yes                                           │         │    │
│  │    _handle_tool_execution()                               │         │    │
│  │           │                                               │         │    │
│  │           ▼                                               │         │    │
│  │    tool_manager.execute_tool("search_course_content")     │         │    │
│  │           │                                               │         │    │
│  │           ▼                                               │         │    │
│  │    search_tools.py: CourseSearchTool.execute()            │         │    │
│  │           │                                               │         │    │
│  │           ▼                                               │         │    │
│  │    vector_store.py: VectorStore.search()                  │         │    │
│  │           │                                               │         │    │
│  │           ▼                                               │         │    │
│  │    ChromaDB query ──► results                             │         │    │
│  │           │                                               │         │    │
│  │           ▼                                               │         │    │
│  │ 2nd Claude API call (without tools) ──► final answer ─────┼─────────┘    │
│  └───────────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tool Calling Pattern (Agentic Loop)

The system uses Claude's tool calling feature with a two-phase approach:

**Phase 1** (`ai_generator.py:79-84`):
- Call Claude API with `tools` and `tool_choice: {"type": "auto"}`
- Claude decides whether to use tools based on the query
- If `response.stop_reason == "tool_use"`, proceed to Phase 2
- If `response.stop_reason == "end_turn"`, return response directly

**Phase 2** (`ai_generator.py:89-135`):
- Extract tool calls from `response.content` (type `tool_use`)
- Execute each tool via `tool_manager.execute_tool(name, **input)`
- Build `tool_result` messages with matching `tool_use_id`
- Call Claude API again WITHOUT tools to get final synthesized answer

### Tool Definition Schema

Tools follow Anthropic's format (`search_tools.py:27-50`):

```python
{
    "name": "search_course_content",
    "description": "...",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "..."},
            "course_name": {"type": "string", "description": "..."},  # optional
            "lesson_number": {"type": "integer", "description": "..."}  # optional
        },
        "required": ["query"]
    }
}
```

### Vector Store Collections

ChromaDB uses two collections (`vector_store.py:51-52`):

| Collection | Purpose | ID Schema |
|------------|---------|-----------|
| `course_catalog` | Course metadata (title, instructor, links, lessons JSON) | `{course_title}` |
| `course_content` | Chunked course text for semantic search | `{course_title}_{chunk_index}` |

**Course name resolution**: When user provides a partial course name, `_resolve_course_name()` does a semantic search against `course_catalog` to find the best matching course title, which is then used to filter `course_content`.

### Data Models (`models.py`)

```
Course
├── title (unique identifier)
├── course_link
├── instructor
└── lessons: List[Lesson]
       ├── lesson_number
       ├── title
       └── lesson_link

CourseChunk (stored in vector DB)
├── content (text)
├── course_title
├── lesson_number
└── chunk_index
```

### Session Management

`SessionManager` (`session_manager.py`) maintains in-memory conversation history:
- Sessions created with `session_{counter}` IDs
- History passed to Claude as formatted string in system prompt
- Limited to `MAX_HISTORY` exchanges (default: 2)

### Document Ingestion Pipeline

On startup (`app.py:88-98`), documents from `../docs` are processed:

```
File (PDF/DOCX/TXT)
    │
    ▼
document_processor.process_course_document()
    │
    ├──► Parse header (Course Title, Link, Instructor)
    ├──► Split by "Lesson N:" markers
    └──► chunk_text() with overlap
           │
           ▼
    List[CourseChunk]
           │
           ▼
vector_store.add_course_metadata(course)
vector_store.add_course_content(chunks)
```

### Document Format

Course documents must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [title]
Lesson Link: [url]
[content]

Lesson 1: [title]
...
```

### Configuration (`config.py`)

| Setting | Default | Purpose |
|---------|---------|---------|
| `ANTHROPIC_MODEL` | claude-sonnet-4-20250514 | Claude model for responses |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer for embeddings |
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `MAX_RESULTS` | 5 | Search results returned |
| `MAX_HISTORY` | 2 | Conversation turns remembered |
| `CHROMA_PATH` | ./chroma_db | Vector DB storage |

Environment: `ANTHROPIC_API_KEY` required in `.env` file.

### Frontend

Static files served by FastAPI (`app.py:119`). Key functions in `script.js`:
- `sendMessage()`: Captures input, calls API, displays response
- `addMessage()`: Renders markdown via `marked.parse()`, shows collapsible sources
- `loadCourseStats()`: Populates sidebar from `/api/courses`

### API Endpoints

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/api/query` | POST | `{query, session_id?}` | `{answer, sources[], session_id}` |
| `/api/courses` | GET | - | `{total_courses, course_titles[]}` |