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
│  │ Sequential Tool Calling Loop (up to 2 rounds)             │         │    │
│  │                                                           │         │    │
│  │ LOOP: while round < MAX_TOOL_ROUNDS (2)                   │         │    │
│  │    │                                                      │         │    │
│  │    ├─► Claude API call (with tools)                       │         │    │
│  │    │      │                                               │         │    │
│  │    │      ├─► stop_reason == "end_turn" ──► return text   │         │    │
│  │    │      │                                               │         │    │
│  │    │      └─► stop_reason == "tool_use"                   │         │    │
│  │    │             │                                        │         │    │
│  │    │             ▼                                        │         │    │
│  │    │      _execute_tools() via tool_manager               │         │    │
│  │    │      ├─► CourseSearchTool → VectorStore.search()     │         │    │
│  │    │      └─► CourseOutlineTool → get_course_outline      │         │    │
│  │    │             │                                        │         │    │
│  │    │             ▼                                        │         │    │
│  │    │      Append tool results to messages                 │         │    │
│  │    │             │                                        │         │    │
│  │    └─────────────┴─► round++ and continue loop            │         │    │
│  │                                                           │         │    │
│  │ After loop: final API call (no tools) ──► answer ─────────┼─────────┘    │
│  └───────────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tool Calling Pattern (Sequential Agentic Loop)

The system uses Claude's tool calling feature with a **sequential loop** supporting up to 2 tool rounds per query. This enables complex queries where one tool's results inform the next search.

**Loop Structure** (`ai_generator.py:51-127`):

```
while round_count < MAX_TOOL_ROUNDS (2):
    1. Call Claude API with tools and tool_choice: {"type": "auto"}
    2. Check stop_reason:
       - "end_turn" or "max_tokens" → extract text and return
       - "tool_use" → execute tools, append results to messages, round++
    3. Continue loop

After loop exhausted → final API call WITHOUT tools to synthesize answer
```

**Key Methods**:
- `generate_response()`: Main entry point with sequential tool calling loop
- `_execute_tools()`: Execute all tool calls, returns (results, has_error)
- `_extract_text_response()`: Handle mixed text/tool_use content blocks
- `_final_call_without_tools()`: Synthesis call after max rounds or errors

**Termination Conditions**:
1. Claude returns `stop_reason == "end_turn"` (natural completion)
2. Maximum rounds reached (`MAX_TOOL_ROUNDS = 2`)
3. Tool execution error (graceful degradation with final call)

**Example Multi-Round Query**:
```
User: "Find courses similar to lesson 3 of the RAG course"
Round 1: Claude calls get_course_outline(course_name="RAG") → gets lesson titles
Round 2: Claude calls search_course_content(query="<lesson 3 title>") → finds similar courses
Final: Claude synthesizes answer from accumulated tool results
```

### Available Tools

The system provides two tools for Claude to use (`search_tools.py`):

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `search_course_content` | Search course materials for specific content | `query` (required), `course_name`, `lesson_number` | Formatted search results with sources |
| `get_course_outline` | Get course structure and lesson list | `course_name` (required) | Course title, link, instructor, and all lessons with links |

### Tool Definition Schema

Tools follow Anthropic's format:

**CourseSearchTool** (`search_tools.py:27-50`):
```python
{
    "name": "search_course_content",
    "description": "Search course materials with smart course name matching and lesson filtering",
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

**CourseOutlineTool** (`search_tools.py:136-151`):
```python
{
    "name": "get_course_outline",
    "description": "Get the complete outline of a course including title, course link, and all lessons",
    "input_schema": {
        "type": "object",
        "properties": {
            "course_name": {"type": "string", "description": "Course title or partial name"}
        },
        "required": ["course_name"]
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

**Course outline retrieval**: `get_course_outline(course_name)` (`vector_store.py:268-306`) returns full course metadata including title, link, instructor, and parsed lesson list with links. Uses semantic matching for partial course names.

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

### Configuration (`config.py` and `ai_generator.py`)

| Setting | Default | Location | Purpose |
|---------|---------|----------|---------|
| `ANTHROPIC_MODEL` | claude-sonnet-4-20250514 | config.py | Claude model for responses |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | config.py | Sentence transformer for embeddings |
| `CHUNK_SIZE` | 800 | config.py | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | config.py | Overlap between chunks |
| `MAX_RESULTS` | 5 | config.py | Search results returned |
| `MAX_HISTORY` | 2 | config.py | Conversation turns remembered |
| `CHROMA_PATH` | ./chroma_db | config.py | Vector DB storage |
| `MAX_TOOL_ROUNDS` | 2 | ai_generator.py | Sequential tool calling rounds per query |

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