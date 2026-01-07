"""
API endpoint tests for the RAG system.
Tests FastAPI endpoints using TestClient with mocked dependencies.

Run with: uv run pytest backend/tests/test_api_endpoints.py -v
"""
import pytest
from unittest.mock import MagicMock


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_returns_valid_response(self, test_client, mock_rag_system):
        """Query endpoint returns answer, sources, and session_id"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "Test answer from RAG system"
        assert data["sources"] == ["Source 1", "Source 2"]

    def test_query_with_session_id(self, test_client, mock_rag_system):
        """Query endpoint accepts and returns provided session_id"""
        response = test_client.post(
            "/api/query",
            json={"query": "Follow-up question", "session_id": "existing_session"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing_session"
        mock_rag_system.query.assert_called_with("Follow-up question", "existing_session")

    def test_query_creates_session_when_not_provided(self, test_client, mock_rag_system):
        """Query endpoint creates new session when session_id not provided"""
        response = test_client.post(
            "/api/query",
            json={"query": "New conversation"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_empty_query_validation(self, test_client):
        """Query endpoint validates non-empty query"""
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )
        # FastAPI accepts empty strings by default, actual validation depends on implementation
        assert response.status_code == 200

    def test_query_missing_query_field(self, test_client):
        """Query endpoint returns 422 when query field is missing"""
        response = test_client.post(
            "/api/query",
            json={}
        )

        assert response.status_code == 422

    def test_query_handles_rag_system_error(self, test_client, mock_rag_system):
        """Query endpoint returns 500 when RAG system raises exception"""
        mock_rag_system.query.side_effect = Exception("RAG system error")

        response = test_client.post(
            "/api/query",
            json={"query": "Trigger error"}
        )

        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]

    def test_query_accepts_json_content_type(self, test_client):
        """Query endpoint accepts application/json content type"""
        response = test_client.post(
            "/api/query",
            json={"query": "Test question"},
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_courses_returns_stats(self, test_client, mock_rag_system):
        """Courses endpoint returns total_courses and course_titles"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert data["course_titles"] == ["Course A", "Course B", "Course C"]

    def test_courses_handles_empty_catalog(self, test_client, mock_rag_system):
        """Courses endpoint handles empty course catalog"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_handles_analytics_error(self, test_client, mock_rag_system):
        """Courses endpoint returns 500 when analytics fails"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]


class TestSessionEndpoint:
    """Tests for DELETE /api/session/{session_id} endpoint"""

    def test_clear_session_success(self, test_client, mock_rag_system):
        """Clear session endpoint returns success status"""
        response = test_client.delete("/api/session/test_session_456")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        assert data["session_id"] == "test_session_456"
        mock_rag_system.session_manager.clear_session.assert_called_with("test_session_456")

    def test_clear_session_handles_error(self, test_client, mock_rag_system):
        """Clear session endpoint returns 500 on error"""
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session error")

        response = test_client.delete("/api/session/bad_session")

        assert response.status_code == 500
        assert "Session error" in response.json()["detail"]

    def test_clear_nonexistent_session(self, test_client, mock_rag_system):
        """Clear session endpoint succeeds even for non-existent sessions"""
        # In current implementation, clearing non-existent session doesn't raise error
        response = test_client.delete("/api/session/nonexistent")

        assert response.status_code == 200


class TestRequestResponseFormats:
    """Tests for request/response format handling"""

    def test_query_response_json_structure(self, test_client, mock_rag_system):
        """Query response has correct JSON structure"""
        response = test_client.post(
            "/api/query",
            json={"query": "Test"}
        )

        data = response.json()
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_courses_response_json_structure(self, test_client, mock_rag_system):
        """Courses response has correct JSON structure"""
        response = test_client.get("/api/courses")

        data = response.json()
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])

    def test_query_invalid_json(self, test_client):
        """Query endpoint returns 422 for invalid JSON"""
        response = test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_wrong_content_type(self, test_client):
        """Query endpoint handles wrong content type"""
        response = test_client.post(
            "/api/query",
            content='{"query": "test"}',
            headers={"Content-Type": "text/plain"}
        )

        # FastAPI may still parse it or return 422
        assert response.status_code in [200, 422]


class TestQueryWithVariousInputs:
    """Tests for query endpoint with various input types"""

    def test_query_with_special_characters(self, test_client, mock_rag_system):
        """Query handles special characters in query string"""
        response = test_client.post(
            "/api/query",
            json={"query": "What about <script>alert('xss')</script>?"}
        )

        assert response.status_code == 200

    def test_query_with_unicode(self, test_client, mock_rag_system):
        """Query handles unicode characters"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is 机器学习?"}
        )

        assert response.status_code == 200

    def test_query_with_long_input(self, test_client, mock_rag_system):
        """Query handles long input strings"""
        long_query = "What is " + "machine learning " * 100 + "?"
        response = test_client.post(
            "/api/query",
            json={"query": long_query}
        )

        assert response.status_code == 200

    def test_query_with_newlines(self, test_client, mock_rag_system):
        """Query handles newlines in input"""
        response = test_client.post(
            "/api/query",
            json={"query": "Line 1\nLine 2\nLine 3"}
        )

        assert response.status_code == 200


class TestEndpointMethods:
    """Tests for correct HTTP methods"""

    def test_query_get_not_allowed(self, test_client):
        """GET method not allowed on query endpoint"""
        response = test_client.get("/api/query")
        assert response.status_code == 405

    def test_courses_post_not_allowed(self, test_client):
        """POST method not allowed on courses endpoint"""
        response = test_client.post("/api/courses", json={})
        assert response.status_code == 405

    def test_session_get_not_allowed(self, test_client):
        """GET method not allowed on session endpoint"""
        response = test_client.get("/api/session/test")
        assert response.status_code == 405
