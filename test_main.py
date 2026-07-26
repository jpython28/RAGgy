import os
import unittest
import uuid
import chromadb
from unittest.mock import patch, Mock, MagicMock
import main
from fastapi.testclient import TestClient

class TestHealth(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(main.app)
    
    def test_get_health(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "server_status": "ok",
            "chroma_status": "ok",
        }

@patch.dict(os.environ, {"API_KEY": "test-key"}, clear=True)
@patch("main.openai_client.chat.completions.create")
class TestQuery(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(main.app)
        self.test_chat = [
            {
                "role": "system",
                "content": "Hello!",
            },
            {
                "role": "user",
                "content": "Hello!",
            },
            {
                "role": "assistant",
                "content": "How can I help you?",
            },
                            {
                "role": "user",
                "content": "I'm just testing!",
            },
        ]
    
    def test_query(self, mock_completion: MagicMock):
        mock_completion.return_value.choices[0].message.content = "test response"
        query = {
            "chat": self.test_chat,
        }
        response = self.client.post("/query", headers={"api-key": "test-key"}, json=query)
        response_chat = response.json()["chat"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response_chat), 5)
        self.assertEqual(response_chat[4]["role"], "assistant")
        self.assertEqual(response_chat[4]["content"], "test response")
        mock_completion.assert_called_once()

    def test_one_message(self, mock_completion: MagicMock):
        mock_completion.return_value.choices[0].message.content = "test response"
        query = {
            "chat": [
                {
                    "role": "user",
                    "content": "Hello!",
                }
            ]
        }
        response = self.client.post("/query", headers={"api-key": "test-key"}, json=query)
        response_chat = response.json()["chat"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response_chat[0]["role"], "system")
        self.assertEqual(response_chat[1]["role"], "user")
        self.assertEqual(response_chat[2]["role"], "assistant")
        mock_completion.assert_called_once()
    
    def test_empty_chat(self, mock_completion: MagicMock):
        query = {"chat": []}
        response = self.client.post("/query", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.status_code, 422)

    def test_nonuser_last_message(self, mock_completion: MagicMock):
        query = {
            "chat": [
                {
                    "role": "assistant",
                    "content": "How can I help you?",
                }
            ]
        }
        response = self.client.post("/query", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.status_code, 422)
    
    def test_empty_last_message(self, mock_completion: MagicMock):
        query = {
            "chat": [
                {
                    "role": "user",
                    "content": "",
                }
            ]
        }
        response = self.client.post("/query", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.status_code, 422)
    
    def test_whitespace_last_message(self, mock_completion: MagicMock):
        query = {
            "chat": [
                {
                    "role": "user",
                    "content": "          ",
                }
            ]
        }
        response = self.client.post("/query", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.status_code, 422)
    
    def test_with_oversized_prompt(self, mock_completion: MagicMock):
        query = {
            "chat": [
                {
                    "role": "user",
                    "content": "x"*5000,
                }
            ]
        }
        response = self.client.post("/query", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.status_code, 422)

    def test_wrong_api_key(self, mock_completion: MagicMock):
        query = {
            "chat": self.test_chat,
        }
        response = self.client.post("/query", headers={"api-key": "wrong-api-key"}, json=query)
        self.assertEqual(response.status_code, 401)

    def test_empty_header(self, mock_completion: MagicMock):
            query = {
                "chat": self.test_chat,
            }
            response = self.client.post("/query", headers={}, json=query)
            self.assertEqual(response.status_code, 401)

    def test_openai_auth_error(self, mock_completion: MagicMock):
        mock_completion.side_effect = main.openai.AuthenticationError("Invalid API key", response=Mock(status_code=401), body=None)
        query = {
            "chat": self.test_chat,
        }
        response = self.client.post("/query", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.status_code, 503)

    def test_openai_rate_limit_error(self, mock_completion: MagicMock):
        mock_completion.side_effect = main.openai.RateLimitError("Rate limit exceeded", response=Mock(status_code=429), body=None)
        query = {
            "chat": self.test_chat,
        }
        response = self.client.post("/query", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.status_code, 429)

    def test_openai_timeout_error(self, mock_completion: MagicMock):
        mock_completion.side_effect = main.openai.APITimeoutError(request=Mock())
        query = {
            "chat": self.test_chat,
        }
        response = self.client.post("/query", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.status_code, 504)

    def test_openai_other_exception(self, mock_completion: MagicMock):
        mock_completion.side_effect = main.openai.APIError(request=Mock(), message="Unknown error", body=None)
        query = {
            "chat": self.test_chat,
        }
        response = self.client.post("/query", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.status_code, 503)

    def test_unrelated_query(self, mock_completion: MagicMock):
        mock_completion.return_value.choices[0].message.content = "test response"
        query = {
            "chat": [
                {
                    "role": "user",
                    "content": "What is the capital of France?",
                }
            ]
        }
        response = self.client.post("/query", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.json()["chunks_used"], 0)
        mock_completion.assert_called_once()

@patch("main.collection", chromadb.EphemeralClient().create_collection("test-documents"))
@patch.dict(os.environ, {"API_KEY": "test-key"}, clear=True)
class TestDocuments(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(main.app)
    
    def test_ingest(self):
        query = {
            "text": str(uuid.uuid4())
        }
        response = self.client.post("/documents", headers={"api-key": "test-key"}, json=query)
        print(response.json())
        self.assertEqual(response.status_code, 201)
    
    def test_empty_document(self):
        query = {
            "text": "",
        }
        response = self.client.post("/documents", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.status_code, 422)

    def test_whitespace_document(self):
        query = {
            "text": "          ",
        }
        response = self.client.post("/documents", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.status_code, 422)

    def test_duplicate_document(self):
        query = {
            "text": str(uuid.uuid4())
        }
        response1 = self.client.post("/documents", headers={"api-key": "test-key"}, json=query)
        response2 = self.client.post("/documents", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response1.status_code, 201)
        self.assertEqual(response2.status_code, 422)

    def test_wrong_api_key(self):
        query = {
            "text": str(uuid.uuid4())
        }
        response = self.client.post("/documents", headers={"api-key": "wrong-api-key"}, json=query)
        self.assertEqual(response.status_code, 401)
    
    def test_empty_header(self):
        query = {
            "text": str(uuid.uuid4())
        }
        response = self.client.post("/documents", headers={}, json=query)
        self.assertEqual(response.status_code, 401)

    def test_oversized_document(self):
        query = {
            "text": "x" * 1_500_000
        }
        response = self.client.post("/documents", headers={"api-key": "test-key"}, json=query)
        self.assertEqual(response.status_code, 422)

if __name__ == '__main__':
    unittest.main()