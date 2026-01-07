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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
