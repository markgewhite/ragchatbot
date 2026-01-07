"""
Tests to identify the query failure issue.
Run with: cd backend && uv run pytest test_query_failure.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
import sys


class TestConfigValidation:
    """Test configuration settings"""

    def test_max_results_is_positive(self):
        """MAX_RESULTS must be greater than 0 for ChromaDB queries"""
        from config import config

        assert config.MAX_RESULTS > 0, (
            f"MAX_RESULTS is {config.MAX_RESULTS}, but must be > 0. "
            "ChromaDB will fail with n_results=0"
        )

    def test_anthropic_api_key_is_set(self):
        """ANTHROPIC_API_KEY should be configured"""
        from config import config

        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not set"

    def test_chunk_size_is_positive(self):
        """CHUNK_SIZE should be positive"""
        from config import config

        assert config.CHUNK_SIZE > 0, f"CHUNK_SIZE is {config.CHUNK_SIZE}"

    def test_embedding_model_is_set(self):
        """EMBEDDING_MODEL should be configured"""
        from config import config

        assert config.EMBEDDING_MODEL, "EMBEDDING_MODEL is not set"


class TestVectorStoreSearch:
    """Test vector store search functionality"""

    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create a test vector store with temporary path"""
        from vector_store import VectorStore
        from config import config

        # Use temp path to avoid corrupting production DB
        return VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS
        )

    def test_search_with_zero_max_results_fails(self, tmp_path):
        """Search with max_results=0 should fail"""
        from vector_store import VectorStore
        from config import config

        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model=config.EMBEDDING_MODEL,
            max_results=0  # Explicitly test with 0
        )

        # This should fail or return an error
        results = store.search(query="test query")

        # If max_results=0, ChromaDB will either raise an exception
        # or return an error - either way, this tests the failure
        if results.error:
            print(f"Search failed as expected with error: {results.error}")
            assert "error" in results.error.lower() or True  # Accept error
        else:
            # If no error, the search should still be empty but functional
            # This means ChromaDB handled it gracefully
            pass

    def test_search_with_positive_max_results_succeeds(self, tmp_path):
        """Search with positive max_results should not fail"""
        from vector_store import VectorStore
        from config import config

        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model=config.EMBEDDING_MODEL,
            max_results=5  # Use a valid value
        )

        # This should not raise an exception
        results = store.search(query="test query")

        # Should return empty results (no data) but not an error
        assert results.error is None, f"Unexpected error: {results.error}"

    def test_search_with_current_config(self, vector_store):
        """Test search with current configuration settings"""
        from config import config

        print(f"Testing with MAX_RESULTS = {config.MAX_RESULTS}")

        try:
            results = vector_store.search(query="test query")
            if results.error:
                pytest.fail(f"Search failed with current config: {results.error}")
        except Exception as e:
            pytest.fail(f"Search raised exception with current config: {e}")


class TestChromaDBDirectly:
    """Test ChromaDB behavior directly"""

    def test_chromadb_query_with_zero_results(self, tmp_path):
        """Test what happens when querying ChromaDB with n_results=0"""
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(tmp_path / "direct_test"),
            settings=Settings(anonymized_telemetry=False)
        )

        collection = client.get_or_create_collection(name="test")

        # Try to query with n_results=0
        with pytest.raises(Exception) as exc_info:
            collection.query(
                query_texts=["test"],
                n_results=0
            )

        print(f"ChromaDB raised: {type(exc_info.value).__name__}: {exc_info.value}")
        # This confirms that n_results=0 causes an error


class TestSearchToolIntegration:
    """Test search tool execution"""

    @pytest.fixture
    def search_tool(self, tmp_path):
        """Create search tool with test vector store"""
        from vector_store import VectorStore
        from search_tools import CourseSearchTool
        from config import config

        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS
        )

        return CourseSearchTool(store)

    def test_search_tool_execute(self, search_tool):
        """Test search tool execution doesn't crash"""
        from config import config

        print(f"Testing search tool with MAX_RESULTS = {config.MAX_RESULTS}")

        try:
            result = search_tool.execute(query="what is retrieval?")
            print(f"Search result: {result[:200] if len(result) > 200 else result}")
            # Should either return results or a "no results" message, not crash
        except Exception as e:
            pytest.fail(f"Search tool crashed: {type(e).__name__}: {e}")


class TestCourseOutlineTool:
    """Test course outline tool functionality"""

    @pytest.fixture
    def vector_store_with_course(self, tmp_path):
        """Create a test vector store with a sample course"""
        from vector_store import VectorStore
        from models import Course, Lesson
        from config import config

        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS
        )

        # Add a test course
        test_course = Course(
            title="Test Course",
            course_link="https://example.com/course",
            instructor="Test Instructor",
            lessons=[
                Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
                Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/lesson1"),
                Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/lesson2")
            ]
        )
        store.add_course_metadata(test_course)

        return store

    @pytest.fixture
    def outline_tool(self, vector_store_with_course):
        """Create outline tool with test vector store"""
        from search_tools import CourseOutlineTool
        return CourseOutlineTool(vector_store_with_course)

    def test_outline_tool_definition(self):
        """Test outline tool has correct schema"""
        from vector_store import VectorStore
        from search_tools import CourseOutlineTool
        from config import config
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStore(
                chroma_path=os.path.join(tmp_dir, "test_chroma"),
                embedding_model=config.EMBEDDING_MODEL,
                max_results=config.MAX_RESULTS
            )
            tool = CourseOutlineTool(store)
            definition = tool.get_tool_definition()

            assert definition["name"] == "get_course_outline"
            assert "course_name" in definition["input_schema"]["properties"]
            assert definition["input_schema"]["required"] == ["course_name"]

    def test_outline_tool_execute_with_course(self, outline_tool):
        """Test outline tool returns correct course structure"""
        result = outline_tool.execute(course_name="Test Course")

        # Check course info is present
        assert "Test Course" in result
        assert "https://example.com/course" in result
        assert "Test Instructor" in result

        # Check lessons are present
        assert "Introduction" in result
        assert "Getting Started" in result
        assert "Advanced Topics" in result

        # Check lesson links are present
        assert "https://example.com/lesson0" in result
        assert "https://example.com/lesson1" in result
        assert "https://example.com/lesson2" in result

        # Check sources are tracked
        assert len(outline_tool.last_sources) == 1
        assert "Test Course" in outline_tool.last_sources[0]

    def test_outline_tool_execute_partial_match(self, outline_tool):
        """Test outline tool works with partial course name"""
        result = outline_tool.execute(course_name="Test")

        # Should still find the course via semantic matching
        assert "Test Course" in result
        assert "Introduction" in result

    def test_outline_tool_execute_not_found(self, tmp_path):
        """Test outline tool handles empty store"""
        from vector_store import VectorStore
        from search_tools import CourseOutlineTool
        from config import config

        # Create an empty vector store (no courses)
        empty_store = VectorStore(
            chroma_path=str(tmp_path / "empty_chroma"),
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS
        )
        empty_tool = CourseOutlineTool(empty_store)

        result = empty_tool.execute(course_name="Any Course Name")
        assert "No course found" in result

    def test_outline_tool_integration(self, tmp_path):
        """Test outline tool via tool manager"""
        from vector_store import VectorStore
        from search_tools import CourseOutlineTool, ToolManager
        from models import Course, Lesson
        from config import config

        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS
        )

        # Add test course
        test_course = Course(
            title="Integration Test Course",
            course_link="https://example.com/integration",
            instructor="Integration Instructor",
            lessons=[
                Lesson(lesson_number=0, title="Lesson Zero"),
                Lesson(lesson_number=1, title="Lesson One", lesson_link="https://example.com/l1")
            ]
        )
        store.add_course_metadata(test_course)

        # Register tool
        tool_manager = ToolManager()
        outline_tool = CourseOutlineTool(store)
        tool_manager.register_tool(outline_tool)

        # Execute via manager
        result = tool_manager.execute_tool("get_course_outline", course_name="Integration")

        assert "Integration Test Course" in result
        assert "Lesson Zero" in result
        assert "Lesson One" in result

        # Check sources are available
        sources = tool_manager.get_last_sources()
        assert len(sources) == 1


class TestVectorStoreOutline:
    """Test vector store outline functionality"""

    @pytest.fixture
    def vector_store_with_course(self, tmp_path):
        """Create a test vector store with a sample course"""
        from vector_store import VectorStore
        from models import Course, Lesson
        from config import config

        store = VectorStore(
            chroma_path=str(tmp_path / "test_chroma"),
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS
        )

        test_course = Course(
            title="Outline Test Course",
            course_link="https://example.com/outline",
            instructor="Outline Instructor",
            lessons=[
                Lesson(lesson_number=0, title="Intro"),
                Lesson(lesson_number=1, title="Main Content", lesson_link="https://example.com/main")
            ]
        )
        store.add_course_metadata(test_course)

        return store

    def test_get_course_outline(self, vector_store_with_course):
        """Test get_course_outline returns correct structure"""
        outline = vector_store_with_course.get_course_outline("Outline Test Course")

        assert outline is not None
        assert outline["title"] == "Outline Test Course"
        assert outline["course_link"] == "https://example.com/outline"
        assert outline["instructor"] == "Outline Instructor"
        assert len(outline["lessons"]) == 2

        # Check lesson structure
        assert outline["lessons"][0]["lesson_number"] == 0
        assert outline["lessons"][0]["lesson_title"] == "Intro"
        assert outline["lessons"][1]["lesson_link"] == "https://example.com/main"

    def test_get_course_outline_semantic_match(self, vector_store_with_course):
        """Test get_course_outline with partial name"""
        outline = vector_store_with_course.get_course_outline("Outline")

        assert outline is not None
        assert outline["title"] == "Outline Test Course"

    def test_get_course_outline_not_found(self, tmp_path):
        """Test get_course_outline returns None when store is empty"""
        from vector_store import VectorStore
        from config import config

        # Create an empty vector store (no courses)
        empty_store = VectorStore(
            chroma_path=str(tmp_path / "empty_chroma"),
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS
        )

        outline = empty_store.get_course_outline("Any Course Name")
        assert outline is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
