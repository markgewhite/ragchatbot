"""
Shared pytest fixtures for RAG system tests.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Add backend directory to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


@dataclass
class TestConfig:
    """Test configuration with safe defaults"""
    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "test-model"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def test_config():
    """Provide test configuration"""
    return TestConfig()


@pytest.fixture
def mock_rag_system():
    """Create a mock RAGSystem for API testing"""
    mock = MagicMock()
    mock.query.return_value = ("Test answer from RAG system", ["Source 1", "Source 2"])
    mock.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course A", "Course B", "Course C"]
    }
    mock.session_manager.create_session.return_value = "test_session_123"
    mock.session_manager.clear_session.return_value = None
    return mock


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for AI generator tests"""
    client = MagicMock()
    return client


@pytest.fixture
def mock_tool_manager():
    """Create a mock tool manager"""
    manager = MagicMock()
    manager.execute_tool.return_value = "Tool execution result"
    manager.get_last_sources.return_value = ["Source from tool"]
    manager.get_tool_definitions.return_value = [
        {"name": "search_course_content", "description": "Search courses"},
        {"name": "get_course_outline", "description": "Get course outline"}
    ]
    return manager


@pytest.fixture
def vector_store_with_test_data(tmp_path, test_config):
    """Create a vector store with sample test data"""
    from vector_store import VectorStore
    from models import Course, Lesson, CourseChunk

    store = VectorStore(
        chroma_path=str(tmp_path / "test_chroma"),
        embedding_model=test_config.EMBEDDING_MODEL,
        max_results=test_config.MAX_RESULTS
    )

    # Add test course metadata
    test_course = Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Test Instructor",
        lessons=[
            Lesson(lesson_number=0, title="Course Overview", lesson_link="https://example.com/ml/0"),
            Lesson(lesson_number=1, title="Linear Regression", lesson_link="https://example.com/ml/1"),
            Lesson(lesson_number=2, title="Neural Networks", lesson_link="https://example.com/ml/2"),
        ]
    )
    store.add_course_metadata(test_course)

    # Add test content chunks
    test_chunks = [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence.",
            course_title="Introduction to Machine Learning",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Linear regression predicts continuous values using a linear equation.",
            course_title="Introduction to Machine Learning",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Neural networks are inspired by biological neural networks.",
            course_title="Introduction to Machine Learning",
            lesson_number=2,
            chunk_index=0
        ),
    ]
    store.add_course_content(test_chunks)

    return store


@pytest.fixture
def test_app(mock_rag_system):
    """
    Create a test FastAPI app without static file mounting.
    This avoids the frontend directory dependency issue.
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI(title="Test RAG API")

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return {"status": "cleared", "session_id": session_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def test_client(test_app):
    """Create a TestClient for the test app"""
    from fastapi.testclient import TestClient
    return TestClient(test_app)


def create_mock_api_response(stop_reason: str, content_blocks: list):
    """Helper to create mock Anthropic API responses"""
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content_blocks
    return response


def create_text_block(text: str):
    """Create a mock text content block"""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def create_tool_use_block(tool_id: str, name: str, input_data: dict):
    """Create a mock tool_use content block"""
    block = MagicMock(spec=['type', 'id', 'name', 'input'])
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data
    return block
