import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from django.test import SimpleTestCase, override_settings
from google.api_core.exceptions import ResourceExhausted

from documents.consumers import ChatConsumer, WorkspaceChatConsumer
from documents.utils.gemini_chat import get_gemini_response


class ChatMessageDuplicateTests(SimpleTestCase):
    databases = []

    def setUp(self):
        import documents.utils.gemini_chat as gemini_chat
        gemini_chat.GEMINI_COOLDOWN_UNTIL = 0.0

        self.user_question = "What is the capital of France?"
        self.mock_doc = MagicMock(id=1, is_processed=True)
        self.mock_chunk = MagicMock(page_number=1, content="Paris is the capital of France.")
        self.prior_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help you?"},
        ]

    async def test_consumer_loads_history_before_saving_message(self):
        """
        Verify ChatConsumer.handle_chat_message loads chat_history BEFORE saving the current message.
        """
        consumer = ChatConsumer()
        consumer.user = MagicMock(id=1, is_authenticated=True, is_premium=False)
        consumer.is_rate_limited = AsyncMock(return_value=MagicMock(limited=False))
        consumer.check_chat_limit = AsyncMock(return_value=False)
        consumer.get_document = AsyncMock(return_value=self.mock_doc)

        mock_session = MagicMock(id=10)
        consumer.get_or_create_session = AsyncMock(return_value=mock_session)

        # Track call order
        call_order = []

        async def fake_get_chat_history(session):
            call_order.append("get_chat_history")
            return list(self.prior_history)

        async def fake_save_user_message(session, content):
            call_order.append("save_user_message")
            return MagicMock(id=101, created_at=MagicMock(isoformat=lambda: "2026-10-29T00:00:00"))

        consumer.get_chat_history = fake_get_chat_history
        consumer.save_user_message = fake_save_user_message
        consumer.send = AsyncMock()
        consumer.process_ai_response = AsyncMock()

        await consumer.handle_chat_message({"content": self.user_question})

        # Assert get_chat_history ran before save_user_message
        self.assertEqual(call_order, ["get_chat_history", "save_user_message"])

        # Assert process_ai_response received chat_history without the current user_question
        _, _, passed_content, passed_history = consumer.process_ai_response.call_args.args
        self.assertEqual(passed_content, self.user_question)
        self.assertNotIn(self.user_question, [m["content"] for m in passed_history])

    async def test_workspace_consumer_loads_history_before_saving_message(self):
        """
        Verify WorkspaceChatConsumer.handle_chat_message loads chat_history BEFORE saving the current message.
        """
        consumer = WorkspaceChatConsumer()
        consumer.user = MagicMock(id=1, is_authenticated=True, is_premium=False)
        consumer.is_rate_limited = AsyncMock(return_value=MagicMock(limited=False))
        consumer.check_chat_limit = AsyncMock(return_value=False)
        consumer.get_workspace = AsyncMock(return_value=MagicMock(id=20))
        consumer.has_any_documents = AsyncMock(return_value=True)
        consumer.get_workspace_documents = AsyncMock(return_value=[self.mock_doc])

        mock_session = MagicMock(id=20)
        consumer.get_or_create_workspace_session = AsyncMock(return_value=mock_session)

        call_order = []

        async def fake_get_chat_history(session):
            call_order.append("get_chat_history")
            return list(self.prior_history)

        async def fake_save_user_message(session, content):
            call_order.append("save_user_message")
            return MagicMock(id=201, created_at=MagicMock(isoformat=lambda: "2026-10-29T00:00:00"))

        consumer.get_chat_history = fake_get_chat_history
        consumer.save_user_message = fake_save_user_message
        consumer.send = AsyncMock()
        consumer.process_workspace_ai_response = AsyncMock()

        await consumer.handle_chat_message({"content": self.user_question})

        self.assertEqual(call_order, ["get_chat_history", "save_user_message"])
        _, _, _, passed_content, passed_history = consumer.process_workspace_ai_response.call_args.args
        self.assertEqual(passed_content, self.user_question)
        self.assertNotIn(self.user_question, [m["content"] for m in passed_history])

    async def test_user_question_appears_once_gemini_path(self):
        """
        (a) Gemini path: assert the user's question appears exactly once in the messages sent to Gemini.
        """
        free_user = MagicMock(is_premium=False)
        mock_model = MagicMock()
        mock_chat = MagicMock()

        sent_messages = []

        async def mock_send(msg, stream=True):
            sent_messages.append(msg)
            async def _gen():
                yield MagicMock(text="Paris is the capital of France.", usage_metadata=None)
            return _gen()

        mock_chat.send_message_async = mock_send
        mock_model.start_chat.return_value = mock_chat

        with patch("documents.utils.gemini_chat._hybrid_retrieve", return_value={"hybrid_chunks": [self.mock_chunk]}), \
             patch("documents.utils.gemini_chat._is_gemini_in_cooldown", return_value=False), \
             patch("google.generativeai.GenerativeModel", return_value=mock_model):

            chunks = []
            async for chunk in get_gemini_response(
                self.user_question,
                self.mock_doc,
                self.prior_history,
                user=free_user,
            ):
                chunks.append(chunk)

        # Inspect history passed to start_chat
        start_chat_call = mock_model.start_chat.call_args
        history_passed = start_chat_call.kwargs.get("history") or start_chat_call.args[0]

        # Count occurrences in history turns
        history_question_count = sum(
            1 for turn in history_passed
            for part in turn.get("parts", [])
            if self.user_question in part
        )

        # Count occurrences in direct send_message_async call
        send_question_count = sum(
            1 for msg in sent_messages
            if self.user_question in str(msg)
        )

        total_gemini_question_count = history_question_count + send_question_count
        self.assertEqual(history_question_count, 0, "User question should not be in prior history")
        self.assertEqual(send_question_count, 1, "User question should be sent once via send_message_async")
        self.assertEqual(total_gemini_question_count, 1, "User question must appear exactly once on Gemini path")

    async def test_user_question_appears_once_deepseek_path(self):
        """
        (b) DeepSeek path: assert the user's question appears exactly once in the messages sent to DeepSeek.
        """
        premium_user = MagicMock(is_premium=True)
        captured_messages = []

        async def mock_stream_deepseek(system_prompt, messages, **kwargs):
            captured_messages.extend(messages)
            yield {"type": "text", "chunk": "Paris is the capital of France."}

        with patch("documents.utils.gemini_chat._hybrid_retrieve", return_value={"hybrid_chunks": [self.mock_chunk]}), \
             patch("documents.utils.deepseek_client.stream_deepseek_chat", side_effect=mock_stream_deepseek):

            chunks = []
            async for chunk in get_gemini_response(
                self.user_question,
                self.mock_doc,
                self.prior_history,
                user=premium_user,
            ):
                chunks.append(chunk)

        question_occurrences = [
            m for m in captured_messages
            if m.get("content") == self.user_question
        ]
        self.assertEqual(
            len(question_occurrences), 1,
            f"User question must appear exactly once in DeepSeek messages, found: {len(question_occurrences)}"
        )

    async def test_user_question_appears_once_groq_fallback_path(self):
        """
        (c) Groq fallback path: assert the user's question appears exactly once in the messages sent to Groq.
        """
        free_user = MagicMock(is_premium=False)
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Paris is the capital of France."}}]
        }

        captured_groq_messages = []

        def mock_requests_post(url, headers=None, json=None, timeout=None):
            if json and "messages" in json:
                captured_groq_messages.extend(json["messages"])
            return mock_response

        with patch("documents.utils.gemini_chat._hybrid_retrieve", return_value={"hybrid_chunks": [self.mock_chunk]}), \
             patch("documents.utils.gemini_chat._is_gemini_in_cooldown", return_value=False), \
             patch("google.generativeai.GenerativeModel.start_chat", side_effect=ResourceExhausted("Rate limit")), \
             patch("requests.post", side_effect=mock_requests_post), \
             override_settings(GROQ_API_KEY="mock-groq-key"):

            chunks = []
            async for chunk in get_gemini_response(
                self.user_question,
                self.mock_doc,
                self.prior_history,
                user=free_user,
            ):
                chunks.append(chunk)

        groq_question_occurrences = [
            m for m in captured_groq_messages
            if m.get("content") == self.user_question
        ]
        self.assertEqual(
            len(groq_question_occurrences), 1,
            f"User question must appear exactly once in Groq payload messages, found: {len(groq_question_occurrences)}"
        )

    async def test_groq_fallback_recovers_from_404_not_found(self):
        """
        Assert that when a model returns 404 Not Found (e.g. deprecated model),
        _call_groq_fallback does NOT crash with HTTPError and tries the next candidate model.
        """
        free_user = MagicMock(is_premium=False)
        models_attempted = []

        resp_404 = MagicMock(status_code=404, text='{"error": {"message": "The model does not exist"}}')
        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = {"choices": [{"message": {"content": "Recovered from fallback model"}}]}

        def mock_requests_post(url, headers=None, json=None, timeout=None):
            model = json.get("model")
            models_attempted.append(model)
            if model == "decommissioned-model":
                return resp_404
            return resp_200

        with patch("documents.utils.gemini_chat._hybrid_retrieve", return_value={"hybrid_chunks": [self.mock_chunk]}), \
             patch("documents.utils.gemini_chat._is_gemini_in_cooldown", return_value=False), \
             patch("google.generativeai.GenerativeModel.start_chat", side_effect=ResourceExhausted("Rate limit")), \
             patch("requests.post", side_effect=mock_requests_post), \
             override_settings(GROQ_API_KEY="mock-groq-key", GROQ_MODEL="decommissioned-model"):

            chunks = []
            async for chunk in get_gemini_response(
                self.user_question,
                self.mock_doc,
                self.prior_history,
                user=free_user,
            ):
                chunks.append(chunk)

        self.assertIn("decommissioned-model", models_attempted)
        self.assertTrue(len(models_attempted) >= 2, "Should attempt fallback model when first returns 404")
        self.assertIn("Recovered from fallback model", "".join(chunks))

    async def test_groq_fallback_returns_friendly_message_on_total_failure(self):
        """
        Assert that if Groq returns 404 for all models or fails, get_gemini_response
        gracefully yields the friendly message instead of raising an uncaught HTTPError.
        """
        from documents.utils.gemini_chat import USER_FRIENDLY_RATE_LIMIT_MSG

        free_user = MagicMock(is_premium=False)
        resp_404 = MagicMock(status_code=404, text='{"error": "Not found"}')

        with patch("documents.utils.gemini_chat._hybrid_retrieve", return_value={"hybrid_chunks": [self.mock_chunk]}), \
             patch("documents.utils.gemini_chat._is_gemini_in_cooldown", return_value=False), \
             patch("google.generativeai.GenerativeModel.start_chat", side_effect=ResourceExhausted("Rate limit")), \
             patch("requests.post", return_value=resp_404), \
             override_settings(GROQ_API_KEY="mock-groq-key"):

            chunks = []
            async for chunk in get_gemini_response(
                self.user_question,
                self.mock_doc,
                self.prior_history,
                user=free_user,
            ):
                chunks.append(chunk)

        self.assertIn(USER_FRIENDLY_RATE_LIMIT_MSG, chunks)
