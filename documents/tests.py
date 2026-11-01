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


@override_settings(ENABLE_AGENTIC_RETRIEVAL=False)
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

            # Lexical branch mocks (both legacy icontains and new SearchVector)
            lexical_mock = MagicMock()
            lexical_mock.__getitem__.side_effect = lambda s: doc_chunks[s]
            lexical_annotate = MagicMock()
            lexical_annotate.order_by.return_value.__getitem__.side_effect = lambda s: doc_chunks[s]
            lexical_mock.annotate.return_value = lexical_annotate
            mock_qs.filter.return_value = lexical_mock
            return mock_qs
        return fake_filter

    @override_settings(ENABLE_RRF_HYBRID_SEARCH=False)
    def test_hybrid_retrieve_flag_off_preserves_legacy_behavior(self):
        """With ENABLE_RRF_HYBRID_SEARCH=False, verify legacy retrieval return shape and ordering."""
        with patch("documents.utils.gemini_chat.DocumentChunk.objects.filter", side_effect=self._get_fake_filter()):
            result = _hybrid_retrieve(self.docs[0], "test message", top_k=5, query_embedding=self.fake_query_embedding)
            self.assertIn("vector_chunks", result)
            self.assertIn("keyword_chunks", result)
            self.assertIn("hybrid_chunks", result)
            self.assertIn("scored", result)
            self.assertEqual(len(result["hybrid_chunks"]), 5)
            self.assertEqual(result["hybrid_chunks"], [r["chunk"] for r in result["scored"]])
            scores = [row["hybrid_score"] for row in result["scored"]]
            self.assertEqual(scores, sorted(scores, reverse=True))

    @override_settings(ENABLE_RRF_HYBRID_SEARCH=True)
    def test_hybrid_retrieve_flag_on_returns_rrf_shape_and_ordering(self):
        """With ENABLE_RRF_HYBRID_SEARCH=True, verify RRF retrieval return shape and ordering."""
        with patch("documents.utils.gemini_chat.DocumentChunk.objects.filter", side_effect=self._get_fake_filter()):
            result = _hybrid_retrieve(self.docs[0], "test message", top_k=5, query_embedding=self.fake_query_embedding)
            self.assertIn("vector_chunks", result)
            self.assertIn("keyword_chunks", result)
            self.assertIn("hybrid_chunks", result)
            self.assertIn("scored", result)
            self.assertEqual(len(result["hybrid_chunks"]), 5)
            self.assertEqual(result["hybrid_chunks"], [r["chunk"] for r in result["scored"]])
            scores = [row["hybrid_score"] for row in result["scored"]]
            self.assertEqual(scores, sorted(scores, reverse=True))
            # Each scored item in RRF path also exposes ranks
            for row in result["scored"]:
                self.assertIn("dense_rank", row)
                self.assertIn("lexical_rank", row)

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

    @override_settings(ENABLE_RRF_HYBRID_SEARCH=False, ENABLE_FLASHRANK_RERANK=False)
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

    def test_rerank_chunks_flashrank_ordering_and_score(self):
        """Verify _rerank_chunks_flashrank orders chunks and assigns rerank_score."""
        from documents.utils.gemini_chat import _rerank_chunks_flashrank
        c1 = MagicMock(id=1, content="Bananas and apples are fruits.")
        c2 = MagicMock(id=2, content="Python is a popular programming language for AI.")
        reranked = _rerank_chunks_flashrank("What programming language is used for artificial intelligence?", [c1, c2], top_n=2)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].id, 2)
        self.assertTrue(hasattr(reranked[0], "rerank_score"))
        self.assertGreater(reranked[0].rerank_score, reranked[1].rerank_score)

    def test_rerank_chunks_flashrank_fallback_on_error(self):
        """Verify fallback to original order if FlashRank raises an exception."""
        from documents.utils.gemini_chat import _rerank_chunks_flashrank
        c1 = MagicMock(id=1, content="Chunk 1")
        c2 = MagicMock(id=2, content="Chunk 2")
        with patch("documents.utils.gemini_chat._get_flashrank_ranker", return_value=None):
            fallback = _rerank_chunks_flashrank("query", [c1, c2], top_n=1)
            self.assertEqual(len(fallback), 1)
            self.assertEqual(fallback[0].id, 1)

    @override_settings(ENABLE_RRF_HYBRID_SEARCH=False, ENABLE_FLASHRANK_RERANK=False, ENABLE_AGENTIC_RETRIEVAL=False)
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


class WorkspaceCreateViewTests(SimpleTestCase):
    databases = []

    def setUp(self):
        from django.test import RequestFactory
        self.factory = RequestFactory()
        self.user = MagicMock(id=1, is_authenticated=True, is_premium=False)

    def test_workspace_create_view_get_renders_modal(self):
        """
        Verify GET /workspaces/create/ returns 200 OK and renders workspaces.html with open_create_modal=True.
        """
        from documents.views_workspace import workspace_create_view
        request = self.factory.get("/workspaces/create/")
        request.user = self.user

        with patch("documents.views_workspace.Workspace.objects.filter") as mock_filter, \
             patch("documents.views_workspace.render") as mock_render:
            mock_filter.return_value.prefetch_related.return_value = []
            mock_render.return_value = MagicMock(status_code=200)

            response = workspace_create_view(request)

            mock_render.assert_called_once()
            _, template_name, context = mock_render.call_args[0]
            self.assertEqual(template_name, "workspaces.html")
            self.assertTrue(context.get("open_create_modal"))
            self.assertEqual(response.status_code, 200)

    def test_workspace_create_view_post_creates_workspace(self):
        """
        Verify POST /workspaces/create/ creates workspace and redirects to workspace_detail.
        """
        from documents.views_workspace import workspace_create_view
        request = self.factory.post("/workspaces/create/", {"name": "New Project", "description": "Notes"})
        request.user = self.user
        request._messages = MagicMock()

        mock_workspace = MagicMock(id=42, name="New Project")
        with patch("documents.views_workspace.Workspace.objects.filter") as mock_filter, \
             patch("documents.views_workspace.Workspace.objects.get_or_create", return_value=(mock_workspace, True)):
            mock_filter.return_value.count.return_value = 0

            response = workspace_create_view(request)

            self.assertEqual(response.status_code, 302)
            self.assertIn("/workspaces/42/", response.url)


class ParentChildRetrievalTests(SimpleTestCase):
    databases = []

    def setUp(self):
        from types import SimpleNamespace
        self.doc = MagicMock(id=1, title="Test Doc", is_processed=True)
        self.parent1 = SimpleNamespace(id=10, page_number=1, content="Parent 1 content spanning multiple sentences.")
        self.parent2 = SimpleNamespace(id=20, page_number=2, content="Parent 2 content discussing other topics.")

        self.child1 = SimpleNamespace(id=101, parent=self.parent1, parent_id=10, page_number=1, content="Child 1 of P1")
        self.child2 = SimpleNamespace(id=102, parent=self.parent1, parent_id=10, page_number=1, content="Child 2 of P1")
        self.child3 = SimpleNamespace(id=103, parent=self.parent2, parent_id=20, page_number=2, content="Child 1 of P2")

    @override_settings(ENABLE_PARENT_CHILD=True, ENABLE_FLASHRANK_RERANK=False, RRF_K=60)
    def test_parent_child_retrieve_deduplication_and_shape(self):
        """Verify _parent_child_retrieve dedupes children by parent and preserves return shape."""
        from documents.utils.gemini_chat import _parent_child_retrieve

        mock_qs = MagicMock()
        # Mock vector search returning child1, child2, child3
        mock_qs.select_related.return_value.annotate.return_value.order_by.return_value = [
            self.child1, self.child2, self.child3
        ]

        with patch("documents.utils.gemini_chat.ChildChunk.objects.filter", return_value=mock_qs), \
             patch("documents.utils.gemini_chat._get_query_embedding", return_value=[0.1] * 768):

            result = _parent_child_retrieve(self.doc, "test query", top_k=5)

            self.assertIn("vector_chunks", result)
            self.assertIn("keyword_chunks", result)
            self.assertIn("hybrid_chunks", result)
            self.assertIn("scored", result)

            # Both child1 and child2 belong to parent1 (id=10), child3 belongs to parent2 (id=20)
            # Resulting parents must be deduplicated to exactly 2 parents
            parents = result["hybrid_chunks"]
            self.assertEqual(len(parents), 2)
            parent_ids = [p.id for p in parents]
            self.assertIn(10, parent_ids)
            self.assertIn(20, parent_ids)

    @override_settings(ENABLE_PARENT_CHILD=True)
    def test_hybrid_retrieve_delegates_to_parent_child_when_children_exist(self):
        """Verify _hybrid_retrieve calls _parent_child_retrieve when flag is on and children exist."""
        from documents.utils.gemini_chat import _hybrid_retrieve

        with patch("documents.utils.gemini_chat.ChildChunk.objects.filter") as mock_filter, \
             patch("documents.utils.gemini_chat._parent_child_retrieve") as mock_pc:
            mock_filter.return_value.exists.return_value = True
            mock_pc.return_value = {"hybrid_chunks": [self.parent1], "scored": []}

            res = _hybrid_retrieve(self.doc, "query")

            mock_pc.assert_called_once()
            self.assertEqual(res["hybrid_chunks"], [self.parent1])

    @override_settings(ENABLE_PARENT_CHILD=False, ENABLE_RRF_HYBRID_SEARCH=True)
    def test_hybrid_retrieve_uses_legacy_rrf_when_flag_off_and_no_children(self):
        """Verify _hybrid_retrieve uses standard RRF when no children exist."""
        from documents.utils.gemini_chat import _hybrid_retrieve

        with patch("documents.utils.gemini_chat.ChildChunk.objects.filter") as mock_filter, \
             patch("documents.utils.gemini_chat._parent_child_retrieve") as mock_pc, \
             patch("documents.utils.gemini_chat._hybrid_retrieve_rrf") as mock_rrf:
            mock_filter.return_value.exists.return_value = False
            mock_pc.return_value = {"hybrid_chunks": [self.parent1], "scored": []}
            mock_rrf.return_value = {"hybrid_chunks": [self.parent1], "scored": []}

            res = _hybrid_retrieve(self.doc, "query")

            mock_pc.assert_not_called()
            mock_rrf.assert_called_once()
            self.assertEqual(res["hybrid_chunks"], [self.parent1])

    @override_settings(ENABLE_PARENT_CHILD=False)
    def test_flag_off_safety_document_with_children_and_null_parent_embeddings(self):
        """
        Verify documents that have ChildChunks and null parent embeddings
        still retrieve correctly even when ENABLE_PARENT_CHILD is False.
        """
        from types import SimpleNamespace
        from documents.utils.gemini_chat import _hybrid_retrieve

        parent_null_emb = SimpleNamespace(id=99, embedding=None, page_number=3, content="Parent text with null embedding")
        child_item = SimpleNamespace(id=901, parent=parent_null_emb, parent_id=99, page_number=3, content="Child text")

        mock_qs = MagicMock()
        mock_qs.select_related.return_value.annotate.return_value.order_by.return_value = [child_item]

        with patch("documents.utils.gemini_chat.ChildChunk.objects.filter") as mock_filter, \
             patch("documents.utils.gemini_chat._get_query_embedding", return_value=[0.1] * 768):
            # exists() is True because this document has children
            mock_filter.return_value.exists.return_value = True
            mock_filter.return_value.select_related = mock_qs.select_related

            res = _hybrid_retrieve(self.doc, "query")

            self.assertEqual(len(res["hybrid_chunks"]), 1)
            self.assertEqual(res["hybrid_chunks"][0].id, 99)
            self.assertIsNone(res["hybrid_chunks"][0].embedding)

    def test_relative_score_cutoff_logic(self):
        """
        Verify _apply_relative_score_cutoff drops items below cutoff (min_relative * top),
        always keeps at least 2, and is disabled when set to 0.
        """
        from documents.utils.gemini_chat import _apply_relative_score_cutoff

        items = [
            {"id": 1, "hybrid_score": 1.0},
            {"id": 2, "hybrid_score": 0.5},
            {"id": 3, "hybrid_score": 0.005},  # below 0.01
            {"id": 4, "hybrid_score": 0.001},  # below 0.01
        ]

        # Standard 0.01 cutoff: top_score = 1.0, cutoff = 0.01
        kept, all_items = _apply_relative_score_cutoff(items, score_key="hybrid_score", min_relative_score=0.01)
        self.assertEqual(len(kept), 2)
        self.assertEqual([x["id"] for x in kept], [1, 2])
        self.assertFalse(all_items[0]["dropped"])
        self.assertFalse(all_items[1]["dropped"])
        self.assertTrue(all_items[2]["dropped"])
        self.assertTrue(all_items[3]["dropped"])

        # Always keep at least 2 even if index 1 is below cutoff
        items_steep = [
            {"id": 1, "hybrid_score": 1.0},
            {"id": 2, "hybrid_score": 0.0001},
            {"id": 3, "hybrid_score": 0.00005},
        ]
        kept_steep, _ = _apply_relative_score_cutoff(items_steep, score_key="hybrid_score", min_relative_score=0.01)
        self.assertEqual(len(kept_steep), 2)
        self.assertEqual([x["id"] for x in kept_steep], [1, 2])

        # Disabled when 0
        kept_disabled, _ = _apply_relative_score_cutoff(items, score_key="hybrid_score", min_relative_score=0.0)
        self.assertEqual(len(kept_disabled), 4)

    def test_parent_child_walks_down_to_unique_parents(self):
        """
        Verify parent selection walks down child candidates to collect RERANK_TOP_N UNIQUE parents.
        """
        from types import SimpleNamespace
        from documents.utils.gemini_chat import _parent_child_retrieve

        p1 = SimpleNamespace(id=1, page_number=1, content="P1 content")
        p2 = SimpleNamespace(id=2, page_number=2, content="P2 content")
        p3 = SimpleNamespace(id=3, page_number=3, content="P3 content")

        # 5 children: C1(P1), C2(P1), C3(P2), C4(P1), C5(P3)
        c1 = SimpleNamespace(id=101, parent=p1, parent_id=1, page_number=1, content="C1 of P1")
        c2 = SimpleNamespace(id=102, parent=p1, parent_id=1, page_number=1, content="C2 of P1")
        c3 = SimpleNamespace(id=103, parent=p2, parent_id=2, page_number=2, content="C3 of P2")
        c4 = SimpleNamespace(id=104, parent=p1, parent_id=1, page_number=1, content="C4 of P1")
        c5 = SimpleNamespace(id=105, parent=p3, parent_id=3, page_number=3, content="C5 of P3")

        mock_qs = MagicMock()
        mock_qs.select_related.return_value.annotate.return_value.order_by.return_value = [c1, c2, c3, c4, c5]

        with patch("documents.utils.gemini_chat.ChildChunk.objects.filter", return_value=mock_qs), \
             patch("documents.utils.gemini_chat._get_query_embedding", return_value=[0.1] * 768), \
             patch("documents.utils.gemini_chat._rerank_chunks_flashrank", side_effect=lambda q, ch, top_n: ch):

            # Request 2 unique parents: should walk down and pick P1 and P2 (skipping C2 which is P1)
            res = _parent_child_retrieve(self.doc, "query", top_k=2)
            self.assertEqual(len(res["hybrid_chunks"]), 2)
            self.assertEqual([p.id for p in res["hybrid_chunks"]], [1, 2])

    def test_workspace_retrieval_mixed_reindexed_and_legacy_documents(self):
        """
        Verify _multi_doc_retrieve handles a mix of reindexed (has children) and legacy (no children) docs,
        applies unique parent selection and cutoff to the merged pool.
        """
        from types import SimpleNamespace
        from documents.utils.gemini_chat import _multi_doc_retrieve

        doc_reindexed = MagicMock(id=10, title="ReindexedDoc", is_processed=True)
        doc_legacy = MagicMock(id=20, title="LegacyDoc", is_processed=True)

        p_reindexed = SimpleNamespace(id=100, page_number=1, content="Reindexed doc parent text")
        c_legacy = SimpleNamespace(id=200, page_number=2, content="Legacy doc chunk text")

        def fake_hybrid_retrieve(doc, query, top_k=5, query_embedding=None):
            if doc.id == 10:
                # Returns parent DocumentChunk
                return {
                    "hybrid_chunks": [p_reindexed],
                    "scored": [{"chunk": p_reindexed, "hybrid_score": 0.85}],
                }
            else:
                # Returns legacy DocumentChunk
                return {
                    "hybrid_chunks": [c_legacy],
                    "scored": [{"chunk": c_legacy, "hybrid_score": 0.001}],  # very low score to test cutoff
                }

        with patch("documents.utils.gemini_chat._hybrid_retrieve", side_effect=fake_hybrid_retrieve), \
             patch("documents.utils.gemini_chat._rerank_chunks_flashrank", side_effect=lambda q, ch, top_n: ch), \
             override_settings(ENABLE_FLASHRANK_RERANK=True, RERANK_MIN_RELATIVE_SCORE=0.01):

            details = _multi_doc_retrieve(
                [doc_reindexed, doc_legacy],
                "query",
                query_embedding=[0.1] * 768,
                return_details=True,
            )

            # Both documents contributed to candidate pool
            all_scored = details["all_scored"]
            self.assertEqual(len(all_scored), 2)
            doc_ids = {item["doc_id"] for item in all_scored}
            self.assertEqual(doc_ids, {10, 20})

            # Both kept because minimum 2 are always kept
            final_items = details["final_items"]
            self.assertEqual(len(final_items), 2)

            context_text = details["context_text"]
            self.assertIn("[ReindexedDoc · Page 1]: Reindexed doc parent text", context_text)
            self.assertIn("[LegacyDoc · Page 2]: Legacy doc chunk text", context_text)


class AgenticRetrievalTests(SimpleTestCase):
    databases = []

    def setUp(self):
        from types import SimpleNamespace
        self.doc1 = SimpleNamespace(id=1, title="Doc 1", is_processed=True)
        self.doc2 = SimpleNamespace(id=2, title="Doc 2", is_processed=True)
        self.doc3 = SimpleNamespace(id=3, title="Doc 3", is_processed=True)
        self.chunk1 = SimpleNamespace(id=101, page_number=1, content="Doc 1 content")
        self.chunk2 = SimpleNamespace(id=102, page_number=2, content="Doc 2 content")
        self.chunk3 = SimpleNamespace(id=103, page_number=3, content="Doc 3 content")

    async def test_direct_route_with_rewrite(self):
        """Verify rewrite_and_route parses direct route and standalone query."""
        import json
        from documents.utils.agentic_retrieval import run_agentic_retrieval

        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "standalone_query": "What are the key findings of the report?",
            "route": "direct",
            "sub_queries": []
        })
        mock_resp.usage_metadata = MagicMock(total_token_count=45)

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_resp)

        history = [{"role": "user", "content": "Tell me about the report"}]

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("documents.utils.agentic_retrieval._hybrid_retrieve", return_value={"hybrid_chunks": [self.chunk1]}):

            result = await run_agentic_retrieval("What are its key findings?", [self.doc1], chat_history=history)

            self.assertEqual(result["route"], "direct")
            self.assertEqual(result["standalone_query"], "What are the key findings of the report?")
            self.assertEqual(result["tokens_used"], 45)
            self.assertIn("Doc 1 content", result["context_text"])

    async def test_decompose_route_and_no_document_starvation(self):
        """
        Verify decompose route runs concurrent retrieval per document and guarantees
        every selected document appears in results (even if not explicitly in sub_queries).
        """
        import json
        from documents.utils.agentic_retrieval import run_agentic_retrieval

        mock_resp = MagicMock()
        # Sub-queries only explicitly mention doc 1 and doc 2, but doc 3 is also selected
        mock_resp.text = json.dumps({
            "standalone_query": "Compare technical requirements across all documents",
            "route": "decompose",
            "sub_queries": [
                {"document_id": 1, "query": "Doc 1 architecture"},
                {"document_id": 2, "query": "Doc 2 architecture"},
            ]
        })
        mock_resp.usage_metadata = MagicMock(total_token_count=80)

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_resp)

        def fake_hybrid_retrieve(doc, query, top_k=2):
            if doc.id == 1:
                return {"hybrid_chunks": [self.chunk1]}
            elif doc.id == 2:
                return {"hybrid_chunks": [self.chunk2]}
            else:
                return {"hybrid_chunks": [self.chunk3]}

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("documents.utils.agentic_retrieval._hybrid_retrieve", side_effect=fake_hybrid_retrieve):

            docs = [self.doc1, self.doc2, self.doc3]
            result = await run_agentic_retrieval("Compare the documents", docs, chat_history=[])

            self.assertEqual(result["route"], "decompose")
            self.assertEqual(result["tokens_used"], 80)

            # Assert every selected document has retrieved chunks (no starvation)
            retrieved = result["retrieved_parents"]
            self.assertIn(1, retrieved)
            self.assertIn(2, retrieved)
            self.assertIn(3, retrieved)
            self.assertEqual(len(retrieved[1]), 1)
            self.assertEqual(len(retrieved[2]), 1)
            self.assertEqual(len(retrieved[3]), 1)

            context = result["context_text"]
            self.assertIn("[Doc 1 · Page 1]: Doc 1 content", context)
            self.assertIn("[Doc 2 · Page 2]: Doc 2 content", context)
            self.assertIn("[Doc 3 · Page 3]: Doc 3 content", context)

    async def test_fallback_on_bad_json_or_exception(self):
        """Verify rewrite_and_route falls back to direct route on malformed JSON or error."""
        from documents.utils.agentic_retrieval import run_agentic_retrieval

        mock_resp = MagicMock()
        mock_resp.text = "NOT JSON RESPONSE"
        mock_resp.usage_metadata = None

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_resp)

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("documents.utils.agentic_retrieval._hybrid_retrieve", return_value={"hybrid_chunks": [self.chunk1]}), \
             patch("documents.utils.agentic_retrieval._generate_shared_query_embedding", return_value=[0.1] * 768), \
             patch("documents.utils.agentic_retrieval._multi_doc_retrieve", return_value={"final_items": [{"doc_id": 1, "chunk": self.chunk1}]}):

            result = await run_agentic_retrieval(
                "Compare things",
                [self.doc1, self.doc2],
                chat_history=[{"role": "user", "content": "Previous topic"}]
            )

            # Falls back to direct with original message
            self.assertEqual(result["route"], "direct")
            self.assertEqual(result["standalone_query"], "Compare things")

    async def test_skip_rule_no_history_single_document(self):
        """Verify LLM call is skipped when no chat history and only one document is selected."""
        from documents.utils.agentic_retrieval import run_agentic_retrieval

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock()

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("documents.utils.agentic_retrieval._hybrid_retrieve", return_value={"hybrid_chunks": [self.chunk1]}):

            result = await run_agentic_retrieval("Explain section 2", [self.doc1], chat_history=[])

            # LLM must not be called
            mock_model.generate_content_async.assert_not_called()
            self.assertEqual(result["route"], "direct")
            self.assertEqual(result["standalone_query"], "Explain section 2")
            self.assertEqual(result["tokens_used"], 0)

    async def test_skip_rule_summary_queries(self):
        """Verify LLM call is skipped for summarize / list all bypass queries."""
        from documents.utils.agentic_retrieval import run_agentic_retrieval

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock()

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("documents.utils.agentic_retrieval._hybrid_retrieve", return_value={"hybrid_chunks": [self.chunk1]}), \
             patch("documents.utils.agentic_retrieval._generate_shared_query_embedding", return_value=[0.1] * 768), \
             patch("documents.utils.agentic_retrieval._multi_doc_retrieve", return_value={"final_items": [{"doc_id": 1, "chunk": self.chunk1}]}):

            result = await run_agentic_retrieval(
                "Please summarize the full document",
                [self.doc1, self.doc2],
                chat_history=[{"role": "user", "content": "Hello"}]
            )

            mock_model.generate_content_async.assert_not_called()
            self.assertEqual(result["route"], "direct")
            self.assertEqual(result["standalone_query"], "Please summarize the full document")
            self.assertEqual(result["tokens_used"], 0)





