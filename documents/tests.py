import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch
from django.test import SimpleTestCase, override_settings
from google.api_core.exceptions import ResourceExhausted

from documents.consumers import ChatConsumer, WorkspaceChatConsumer
from documents.utils.gemini_chat import (
    get_gemini_response,
    get_workspace_response,
    _hybrid_retrieve,
    _multi_doc_retrieve,
)


class ChatMessageDuplicateTests(SimpleTestCase):
    databases = []

    def setUp(self):
        import documents.utils.gemini_chat as gemini_chat
        gemini_chat.GEMINI_COOLDOWN_UNTIL = 0.0

        self.user_question = "What is the capital of France?"
        self.mock_doc = MagicMock(id=1, is_processed=True, title="TestDoc")
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

    async def test_groq_fallback_in_cooldown_receives_document_text(self):
        """
        (i) In cooldown, Groq receives system_parts that contain document text.
        """
        free_user = MagicMock(is_premium=False)
        fallback_doc_text = "[Page 1]: Crucial document context about France."

        with patch("documents.utils.gemini_chat._is_gemini_in_cooldown", return_value=True), \
             patch("documents.utils.gemini_chat._get_document_fallback_context", return_value=fallback_doc_text), \
             patch("documents.utils.gemini_chat._call_groq_fallback", return_value="Groq answer") as mock_groq:

            chunks = []
            async for chunk in get_gemini_response(
                self.user_question,
                self.mock_doc,
                self.prior_history,
                user=free_user,
            ):
                chunks.append(chunk)

        mock_groq.assert_called_once()
        system_parts = mock_groq.call_args[0][1]
        self.assertTrue(
            any(fallback_doc_text in part for part in system_parts),
            f"Groq system_parts should contain document text, got: {system_parts}"
        )
        self.assertIn("Groq answer", "".join(chunks))

    async def test_groq_fallback_in_outer_exception_receives_document_text(self):
        """
        (ii) In the outer-exception path, Groq receives document text.
        """
        free_user = MagicMock(is_premium=False)
        fallback_doc_text = "[Page 1]: Fallback text recovered during exception."

        # Cause an unexpected outer exception during generation
        with patch("documents.utils.gemini_chat._is_gemini_in_cooldown", return_value=False), \
             patch("documents.utils.gemini_chat._hybrid_retrieve", side_effect=RuntimeError("Retrieval database crashed")), \
             patch("documents.utils.gemini_chat._get_document_fallback_context", return_value=fallback_doc_text), \
             patch("google.generativeai.GenerativeModel.start_chat", side_effect=RuntimeError("Model crashed")), \
             patch("documents.utils.gemini_chat._call_groq_fallback", return_value="Groq answer") as mock_groq:

            chunks = []
            async for chunk in get_gemini_response(
                self.user_question,
                self.mock_doc,
                self.prior_history,
                user=free_user,
            ):
                chunks.append(chunk)

        mock_groq.assert_called_once()
        system_parts = mock_groq.call_args[0][1]
        self.assertTrue(
            any(fallback_doc_text in part for part in system_parts),
            f"Groq system_parts in outer exception should contain document text, got: {system_parts}"
        )
        self.assertIn("Groq answer", "".join(chunks))

    async def test_groq_not_called_when_document_has_no_chunks(self):
        """
        (iii) When the document has no chunks, Groq is not called and the user gets the 'could not read' message.
        """
        free_user = MagicMock(is_premium=False)

        with patch("documents.utils.gemini_chat._is_gemini_in_cooldown", return_value=True), \
             patch("documents.utils.gemini_chat._get_document_fallback_context", return_value=None), \
             patch("documents.utils.gemini_chat._call_groq_fallback") as mock_groq:

            chunks = []
            async for chunk in get_gemini_response(
                self.user_question,
                self.mock_doc,
                self.prior_history,
                user=free_user,
            ):
                chunks.append(chunk)

        mock_groq.assert_not_called()
        combined = "".join(chunks).lower()
        self.assertTrue(
            "could not be read" in combined or "could not read" in combined,
            f"Expected 'could not read' error message, got: {combined}"
        )

    async def test_workspace_groq_fallback_in_cooldown_receives_document_text(self):
        """
        Assert workspace cooldown branch passes document contexts labeled [Doc title · Page X] to Groq.
        """
        free_user = MagicMock(is_premium=False)
        workspace_doc_text = "[Report · Page 1]: Revenue increased 20%."

        with patch("documents.utils.gemini_chat._is_gemini_in_cooldown", return_value=True), \
             patch("documents.utils.gemini_chat._get_workspace_fallback_context", return_value=workspace_doc_text), \
             patch("documents.utils.gemini_chat._call_groq_fallback", return_value="Workspace groq reply") as mock_groq:

            chunks = []
            async for chunk in get_workspace_response(
                self.user_question,
                [self.mock_doc],
                self.prior_history,
                user=free_user,
            ):
                chunks.append(chunk)

        mock_groq.assert_called_once()
        system_parts = mock_groq.call_args[0][1]
        self.assertTrue(
            any(workspace_doc_text in part for part in system_parts),
            f"Groq system_parts in workspace cooldown should contain labeled text, got: {system_parts}"
        )
        self.assertIn("Workspace groq reply", "".join(chunks))

    async def test_workspace_not_called_when_no_chunks(self):
        """
        Assert workspace cooldown branch does not call Groq when documents have no chunks.
        """
        free_user = MagicMock(is_premium=False)

        with patch("documents.utils.gemini_chat._is_gemini_in_cooldown", return_value=True), \
             patch("documents.utils.gemini_chat._get_workspace_fallback_context", return_value=None), \
             patch("documents.utils.gemini_chat._call_groq_fallback") as mock_groq:

            chunks = []
            async for chunk in get_workspace_response(
                self.user_question,
                [self.mock_doc],
                self.prior_history,
                user=free_user,
            ):
                chunks.append(chunk)

        mock_groq.assert_not_called()
        combined = "".join(chunks).lower()
        self.assertTrue(
            "could not be read" in combined or "could not read" in combined,
            f"Expected 'could not read' error message, got: {combined}"
        )


class MultiDocRetrievalMergeTests(SimpleTestCase):
    databases = []

    def setUp(self):
        self.fake_query_embedding = [1.0] * 768

        class MockChunk:
            def __init__(self, id, doc, content, page_number, dist_val):
                self.id = id
                self.document = doc
                self.content = content
                self.page_number = page_number
                self.embedding = [1.0 - (dist_val / math.sqrt(768))] * 768
                self.distance = None

        self.MockChunk = MockChunk
        self.docs = [
            MagicMock(id=1, is_processed=True, title="Doc 1"),
            MagicMock(id=2, is_processed=True, title="Doc 2"),
            MagicMock(id=3, is_processed=True, title="Doc 3"),
            MagicMock(id=4, is_processed=True, title="Doc 4"),
        ]

        # 4 documents with 5 chunks each (20 chunks total).
        # Doc 4 Chunk 0 has distance 0.0 -> highest hybrid score across all documents.
        # Doc 1 chunks have distances 1.0 - 1.4
        # Doc 2 chunks have distances 2.0 - 2.4
        # Doc 3 chunks have distances 3.0 - 3.4
        # Doc 4 chunks 1-4 have distances 4.1 - 4.4
        self.chunks_by_doc = {
            1: [MockChunk(100 + i, self.docs[0], f"D1 chunk {i}", i + 1, 1.0 + 0.1 * i) for i in range(5)],
            2: [MockChunk(200 + i, self.docs[1], f"D2 chunk {i}", i + 1, 2.0 + 0.1 * i) for i in range(5)],
            3: [MockChunk(300 + i, self.docs[2], f"D3 chunk {i}", i + 1, 3.0 + 0.1 * i) for i in range(5)],
            4: [
                MockChunk(400, self.docs[3], "Doc 4 best chunk content", 1, 0.0),
                MockChunk(401, self.docs[3], "D4 chunk 1", 2, 4.1),
                MockChunk(402, self.docs[3], "D4 chunk 2", 3, 4.2),
                MockChunk(403, self.docs[3], "D4 chunk 3", 4, 4.3),
                MockChunk(404, self.docs[3], "D4 chunk 4", 5, 4.4),
            ],
        }

    def _get_fake_filter(self):
        def fake_filter(document=None, **kwargs):
            doc_chunks = self.chunks_by_doc[document.id]
            mock_qs = MagicMock()
            def fake_annotate(**ann_kwargs):
                for c in doc_chunks:
                    diff_sq = sum((a - b)**2 for a, b in zip(c.embedding, self.fake_query_embedding))
                    c.distance = math.sqrt(diff_sq)
                sorted_c = sorted(doc_chunks, key=lambda x: x.distance)
                res_mock = MagicMock()
                res_mock.order_by.return_value.__getitem__.side_effect = lambda s: sorted_c[s]
                return res_mock
            mock_qs.annotate.side_effect = fake_annotate
            mock_qs.filter.return_value.__getitem__.side_effect = lambda s: []
            return mock_qs
        return fake_filter

    def test_hybrid_retrieve_preserves_original_keys_and_adds_scored(self):
        """
        (d) Assert _hybrid_retrieve's original three keys (vector_chunks, keyword_chunks, hybrid_chunks)
        are returned unchanged for single-document queries, alongside the additive 'scored' key.
        """
        with patch("documents.utils.gemini_chat.DocumentChunk.objects.filter", side_effect=self._get_fake_filter()):
            result = _hybrid_retrieve(self.docs[0], "test message", top_k=5, query_embedding=self.fake_query_embedding)

            self.assertIn("vector_chunks", result)
            self.assertIn("keyword_chunks", result)
            self.assertIn("hybrid_chunks", result)
            self.assertIn("scored", result)

            self.assertEqual(len(result["hybrid_chunks"]), 5)
            self.assertEqual(len(result["scored"]), 5)

            # Assert scored items contain chunk and hybrid_score
            for row in result["scored"]:
                self.assertIn("chunk", row)
                self.assertIn("hybrid_score", row)

            # Assert hybrid_chunks aligns with scored chunks
            scored_chunks = [row["chunk"] for row in result["scored"]]
            self.assertEqual(result["hybrid_chunks"], scored_chunks)

            # Assert scored is sorted descending by hybrid_score
            scores = [row["hybrid_score"] for row in result["scored"]]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_multi_doc_retrieve_merge_sort_and_limit(self):
        """
        Assert:
        (a) a chunk from the 4th document with the best score is present in the context;
        (b) at most 15 chunks are returned;
        (c) chunks are ordered by score descending.
        """
        with patch("documents.utils.gemini_chat.DocumentChunk.objects.filter", side_effect=self._get_fake_filter()):
            context = _multi_doc_retrieve(self.docs, "test query", query_embedding=self.fake_query_embedding)
            self.assertIsNotNone(context)

            chunk_blocks = context.split("\n\n")

            # (b) at most 15 chunks are returned
            self.assertLessEqual(len(chunk_blocks), 15)
            self.assertEqual(len(chunk_blocks), 15)

            # (a) chunk from the 4th document with the best score is present in the context
            self.assertIn("Doc 4 best chunk content", context)
            self.assertTrue(chunk_blocks[0].startswith("[Doc 4 · Page 1]: Doc 4 best chunk content"))

            # (c) chunks are ordered by score descending:
            # Rank 1: Doc 4 chunk 0 (distance 0.0 -> score ~0.75)
            # Ranks 2-6: Doc 1 chunks 0..4 (distances 1.0..1.4)
            # Ranks 7-11: Doc 2 chunks 0..4 (distances 2.0..2.4)
            # Ranks 12-15: Doc 3 chunks 0..3 (distances 3.0..3.3)
            # Ranks 16-20 (excluded): Doc 3 chunk 4 and Doc 4 chunks 1..4
            self.assertIn("Doc 4 best chunk content", chunk_blocks[0])
            for i in range(1, 6):
                self.assertIn(f"D1 chunk {i - 1}", chunk_blocks[i])
            for i in range(6, 11):
                self.assertIn(f"D2 chunk {i - 6}", chunk_blocks[i])
            for i in range(11, 15):
                self.assertIn(f"D3 chunk {i - 11}", chunk_blocks[i])

            # Chunks from rank 16-20 should not be present
            self.assertNotIn("D3 chunk 4", context)
            self.assertNotIn("D4 chunk 1", context)

    async def test_get_workspace_response_integration_includes_top_scored_chunk(self):
        """
        Assert get_workspace_response passes the globally ranked top-15 context to the model prompt.
        """
        mock_model = MagicMock()
        mock_chat = MagicMock()

        async def mock_send(msg, stream=True):
            async def _gen():
                yield MagicMock(text="Workspace answer", usage_metadata=None)
            return _gen()

        mock_chat.send_message_async = mock_send
        mock_model.start_chat.return_value = mock_chat

        with patch("documents.utils.gemini_chat.DocumentChunk.objects.filter", side_effect=self._get_fake_filter()), \
             patch("google.generativeai.GenerativeModel", return_value=mock_model):

            streamed = []
            async for chunk in get_workspace_response(
                "compare the documents",
                self.docs,
                [],
                query_embedding=self.fake_query_embedding,
            ):
                streamed.append(chunk)

            self.assertIn("Workspace answer", streamed)
            start_chat_call = mock_model.start_chat.call_args
            history = start_chat_call.kwargs.get("history") or start_chat_call.args[0]
            system_instruction = history[0]["parts"][0]

            # Assert Doc 4's best chunk is included in the prompt context
            self.assertIn("[Doc 4 · Page 1]: Doc 4 best chunk content", system_instruction)

