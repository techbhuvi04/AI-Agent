from unittest.mock import MagicMock, patch

from recon import llm_client


class TestGetClient:
    def test_returns_none_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            assert llm_client.get_client() is None

    def test_returns_wrapped_client_with_api_key(self):
        raw_client = MagicMock()
        with patch.dict("os.environ", {"GROQ_API_KEY": "fake"}, clear=True):
            with patch("groq.Groq", return_value=raw_client):
                client = llm_client.get_client()
        assert client is not None

    def test_is_available_reflects_env(self):
        with patch.dict("os.environ", {}, clear=True):
            assert llm_client.is_available() is False
        with patch.dict("os.environ", {"GROQ_API_KEY": "x"}, clear=True):
            assert llm_client.is_available() is True


class TestCall:
    def test_generate_content_returns_response_text(self):
        raw_response = MagicMock()
        raw_response.choices = [MagicMock(message=MagicMock(content="the answer"))]
        raw_client = MagicMock()
        raw_client.chat.completions.create.return_value = raw_response

        with patch.dict("os.environ", {"GROQ_API_KEY": "fake"}, clear=True):
            with patch("groq.Groq", return_value=raw_client):
                client = llm_client.get_client()
                text = llm_client.call(client, "what is 2+2?")

        assert text == "the answer"
        _, kwargs = raw_client.chat.completions.create.call_args
        assert kwargs["messages"] == [{"role": "user", "content": "what is 2+2?"}]

    def test_call_returns_none_on_exception(self):
        client = MagicMock()
        client.generate_content.side_effect = RuntimeError("boom")
        assert llm_client.call(client, "prompt") is None
