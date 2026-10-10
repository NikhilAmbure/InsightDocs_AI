import asyncio
import json
import logging

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.conf import settings
from .models import Document, ChatSession, ChatMessage
from .utils.gemini_chat import get_gemini_response


logger = logging.getLogger(__name__)

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """Handle WebSocket connection"""
        self.user = self.scope["user"]
        self.document_id = self.scope['url_route']['kwargs']['document_id']
        
        if not self.user.is_authenticated:
            await self.close()
            return

        self.room_group_name = f"chat_{self.document_id}_{self.user.id}"

        # Check permission
        has_permission = await self.check_document_permission()
        if not has_permission:
            await self.close()
            return

        # Join room group
        if self.channel_layer is not None:
            await self.channel_layer.group_add(
                self.room_group_name,
                self.channel_name
            )
        
        await self.accept()
        logger.info(f"WebSocket connected: {self.user.username} - Doc {self.document_id}")

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        if hasattr(self, "room_group_name") and self.channel_layer is not None:
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )

        logger.info(f"WebSocket disconnected: {self.user} - Code {close_code}")

    async def receive(self, text_data):
        """Receive message from WebSocket"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'chat_message':
                await self.handle_chat_message(data)
            elif message_type == 'typing':
                await self.handle_typing(data)
                
        except json.JSONDecodeError:
            await self.send_error("Invalid JSON format")
        except Exception as e:
            logger.error(f"Error in receive: {str(e)}", exc_info=True)
            await self.send_error("An unexpected error occurred.")

    async def handle_chat_message(self, data):
        """Handle incoming chat message"""
        content = data.get('content', '').strip()
        
        if not content:
            return
        
        # Check Rate Limit
        rate_limit_result = await self.is_rate_limited()
        if rate_limit_result.limited:
            await self.send_error(f"Rate limit exceeded. Please wait {rate_limit_result.retry_after} seconds.")
            return

        # Check Limits
        limit_reached = await self.check_chat_limit()
        if limit_reached:
            await self.send_error("You have reached your free plan chat limit of 500 messages. Please upgrade to Pro (₹199/mo) or Industry (₹1,999/mo) to continue.")
            return

        # Get Document & Session
        document = await self.get_document()
        if not document:
            await self.send_error("Document not found")
            return

        if not document.is_processed:
            await self.send_error("The document is still being analyzed and indexed. Please wait a moment until it's ready.")
            return

        session = await self.get_or_create_session(document)

        # Save user message
        user_msg = await self.save_user_message(session, content)

        # Send acknowledgment to client
        await self.send(text_data=json.dumps({
            'type': 'user_message',
            'id': user_msg.id,
            'content': content,
            'timestamp': user_msg.created_at.isoformat()
        }))

        # Get chat history
        chat_history = await self.get_chat_history(session)

        # Process with Gemini / DeepSeek
        await self.process_ai_response(document, session, content, chat_history)

    async def process_ai_response(self, document, session, user_message, chat_history):
        """Process message through Gemini AI or DeepSeek AI using RAG (Streaming)"""
        try:
            # Check user token limit before calling Gemini utility
            is_limit_reached = await self.check_token_balance_status()
            if is_limit_reached:
                await self.send_error("Token limit reached. Please upgrade to continue.")
                return

            if self.user.is_premium:
                has_quota = await self.has_token_quota()
                if not has_quota:
                    await self.send_error(
                        "You've used up your Pro plan quota for this billing cycle. "
                        "It renews on your next billing date."
                    )
                    return
            
            # 1. Notify client that AI is thinking
            active_model = "DeepSeek v4 Flash" if self.user.is_premium else "Gemini 2.5 Flash"
            await self.send(text_data=json.dumps({
                'type': 'ai_thinking',
                'model': active_model
            }))

            full_response = ""

            async def _record_usage(total_tokens):
                usage = await self.deduct_user_tokens(total_tokens)
                await self.send(text_data=json.dumps({
                    'type': 'usage_update',
                    'usage': usage
                }))

            # 2. Stream chunks to the client as we receive them
            async for chunk in get_gemini_response(
                user_message,
                document,
                chat_history,
                usage_callback=_record_usage,
                user=self.user,
            ):
                if isinstance(chunk, dict) and chunk.get("type") == "thinking":
                    th_text = chunk.get("chunk", "")
                    if th_text:
                        await self.send(text_data=json.dumps({
                            'type': 'ai_thinking_chunk',
                            'content': th_text
                        }))
                else:
                    text_chunk = chunk.get("chunk", "") if isinstance(chunk, dict) else chunk
                    if text_chunk:
                        full_response += text_chunk
                        await self.send(text_data=json.dumps({
                            'type': 'ai_stream_chunk',
                            'content': text_chunk
                        }))

            # 3. Save AI Message to DB
            ai_msg = await self.save_ai_message(session, full_response)

            # 4. Finalize stream on client with full message + metadata
            await self.send(text_data=json.dumps({
                'type': 'ai_stream_end',
                'content': full_response,
                'id': ai_msg.id,
                'timestamp': ai_msg.created_at.isoformat()
            }))

        except Exception as e:
            logger.error(f"RAG Processing Error: {e}", exc_info=True)
            await self.send_error("I encountered an error reading the document context.")

    async def handle_typing(self, data):
        """Broadcast typing indicator"""
        if self.channel_layer:
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'typing_indicator',
                    'user': self.user.username
                }
            )

    async def typing_indicator(self, event):
        """Receive typing event from group"""
        if event['user'] != self.user.username:
            await self.send(text_data=json.dumps({
                'type': 'user_typing',
                'user': event['user']
            }))

    async def send_error(self, message):
        """Send error message to client"""
        await self.send(text_data=json.dumps({
            'type': 'error',
            'message': message
        }))

    # --- Database Operations (Wrapped in sync_to_async) ---
    @database_sync_to_async
    def check_document_permission(self):
        try:
            Document.objects.get(id=self.document_id, owner=self.user)
            return True
        except Document.DoesNotExist:
            return False

    @database_sync_to_async
    def is_rate_limited(self):
        from .utils.rate_limit import check_user_rate_limit
        # Free users: 10 messages per min; Premium users: 60 messages per min
        limit = 60 if self.user.is_premium else 10
        window = 60
        return check_user_rate_limit(self.user, "chat_ws", limit, window)

    @database_sync_to_async
    def check_token_balance_status(self):
        from payments.models import Subscription
        try:
            sub = Subscription.objects.get(user=self.user)
            if sub.is_token_metered:
                return sub.tokens_allocated <= 0
        except Subscription.DoesNotExist:
            pass
        return False

    @database_sync_to_async
    def has_token_quota(self):
        from payments.services import has_available_tokens
        return has_available_tokens(self.user)

    @database_sync_to_async
    def deduct_user_tokens(self, total_tokens):
        from payments.services import deduct_tokens, get_usage_snapshot
        deduct_tokens(self.user, total_tokens)
        return get_usage_snapshot(self.user)

    @database_sync_to_async
    def get_document(self):
        try:
            return Document.objects.get(id=self.document_id, owner=self.user)
        except Document.DoesNotExist:
            return None

    @database_sync_to_async
    def get_or_create_session(self, document):
        session, _ = ChatSession.objects.get_or_create(document=document, user=self.user)
        return session

    @database_sync_to_async
    def save_user_message(self, session, content):
        return ChatMessage.objects.create(session=session, role="user", content=content)

    @database_sync_to_async
    def save_ai_message(self, session, content):
        return ChatMessage.objects.create(session=session, role="assistant", content=content)

    @database_sync_to_async
    def get_chat_history(self, session, limit=20):
        # Rolling window of the last 20 messages prevents latency bloat and context overflow
        messages = list(
            ChatMessage.objects.filter(session=session)
            .order_by('-created_at')[:limit]
        )
        messages.reverse()
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    @database_sync_to_async
    def check_chat_limit(self):
        # Pro and Industry subscribers are metered by token quotas, bypassing global message table scans
        if getattr(self.user, "is_premium", False):
            return False
        user_msg_count = ChatMessage.objects.filter(
            session__user=self.user, 
            role='user'
        ).count()
        return user_msg_count >= 500


# ─────────────────────────────────────────────────────────────────────
#  Feature 4: AI Document Editor Consumer
# ─────────────────────────────────────────────────────────────────────

class EditorConsumer(AsyncWebsocketConsumer):
    """
    Handles real-time AI editing actions (rewrite, summarize, expand, tone adjust)
    for the rich text editor. Streams results back to the client.
    """
    async def connect(self):
        self.user = self.scope["user"]
        self.document_id = self.scope['url_route']['kwargs']['document_id']
        
        if not self.user.is_authenticated:
            await self.close()
            return
            
        has_permission = await self.check_document_permission()
        if not has_permission:
            await self.close()
            return

        await self.accept()
        logger.info(f"Editor WS connected: {self.user.username} - Doc {self.document_id}")

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            action = data.get("action")
            text = data.get("text", "").strip()
            
            if not text:
                return
                
            await self.process_editor_action(action, text, data)
                
        except json.JSONDecodeError:
            await self.send_error("Invalid JSON format")
        except Exception as e:
            logger.error(f"Editor error: {e}", exc_info=True)
            await self.send_error("Editor error occurred.")

    async def process_editor_action(self, action, text, data):
        """Builds prompt based on action and streams Gemini response."""
        # 1. Rate limit check
        rate_limit_result = await self.is_rate_limited()
        if rate_limit_result.limited:
            await self.send_error(f"Rate limit exceeded. Please wait {rate_limit_result.retry_after} seconds.")
            return

        # 2. Token status checks
        is_limit_reached = await self.check_token_balance_status()
        if is_limit_reached:
            await self.send_error("Token limit reached. Please upgrade to continue.")
            return

        if self.user.is_premium:
            has_quota = await self.has_token_quota()
            if not has_quota:
                await self.send_error(
                    "You've used up your Pro plan quota for this billing cycle. "
                    "It renews on your next billing date."
                )
                return

        prompt = ""
        if action == "rewrite":
            prompt = f"Rewrite the following text to make it clearer and more professional. Return ONLY the rewritten text:\n\n{text}"
        elif action == "summarize":
            prompt = f"Summarize the following text concisely. Return ONLY the summary:\n\n{text}"
        elif action == "expand":
            prompt = f"Expand the following text with more detail and context while maintaining the original meaning. Return ONLY the expanded text:\n\n{text}"
        elif action == "adjust_tone":
            tone = data.get("tone", "professional")
            prompt = f"Rewrite the following text in a {tone} tone. Return ONLY the rewritten text:\n\n{text}"
        else:
            await self.send_error("Unknown editor action.")
            return

        try:
            await self.send(text_data=json.dumps({"type": "ai_thinking"}))
            
            # Simple wrapper to async query Gemini stream
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            # We run the blocking IO in a thread
            response_stream = await asyncio.to_thread(
                model.generate_content, 
                prompt, 
                stream=True
            )
            
            full_response = ""
            last_usage = None
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    await self.send(text_data=json.dumps({
                        "type": "ai_stream_chunk",
                        "content": chunk.text,
                        "action": action
                    }))
                last_usage = getattr(chunk, "usage_metadata", None) or last_usage
                    
            await self.send(text_data=json.dumps({
                "type": "ai_stream_end",
                "content": full_response,
                "action": action
            }))

            # Deduct tokens after successful response generation
            if last_usage is not None:
                prompt_tokens = getattr(last_usage, "prompt_token_count", 0) or 0
                candidates_tokens = getattr(last_usage, "candidates_token_count", 0) or 0
                total_tokens = prompt_tokens + candidates_tokens
                if total_tokens > 0:
                    usage = await self.deduct_user_tokens(total_tokens)
                    await self.send(text_data=json.dumps({
                        'type': 'usage_update',
                        'usage': usage
                    }))
            
        except ResourceExhausted:
            logger.warning(f"Gemini API rate limit exceeded in editor: {action}")
            await self.send_error("API Limit Reached. The AI service is currently experiencing high demand. Please try again in a few moments.")
        except Exception as e:
            logger.error(f"Editor AI Error: {e}", exc_info=True)
            await self.send_error("Failed to process text with AI. Please try again later.")

    async def send_error(self, message):
        await self.send(text_data=json.dumps({"type": "error", "message": message}))

    # --- Database Operations (Wrapped in sync_to_async) ---
    @database_sync_to_async
    def check_document_permission(self):
        try:
            Document.objects.get(id=self.document_id, owner=self.user)
            return True
        except Document.DoesNotExist:
            return False

    @database_sync_to_async
    def is_rate_limited(self):
        from .utils.rate_limit import check_user_rate_limit
        # Free users: 10 editor requests per min; Premium users: 60 editor requests per min
        limit = 60 if self.user.is_premium else 10
        window = 60
        return check_user_rate_limit(self.user, "editor_ws", limit, window)

    @database_sync_to_async
    def check_token_balance_status(self):
        from payments.models import Subscription
        try:
            sub = Subscription.objects.get(user=self.user)
            if sub.is_token_metered:
                return sub.tokens_allocated <= 0
        except Subscription.DoesNotExist:
            pass
        return False

    @database_sync_to_async
    def has_token_quota(self):
        from payments.services import has_available_tokens
        return has_available_tokens(self.user)

    @database_sync_to_async
    def deduct_user_tokens(self, total_tokens):
        from payments.services import deduct_tokens, get_usage_snapshot
        deduct_tokens(self.user, total_tokens)
        return get_usage_snapshot(self.user)


# ─────────────────────────────────────────────────────────────────────
#  Multi-Document Workspace Chat Consumer
# ─────────────────────────────────────────────────────────────────────

class WorkspaceChatConsumer(AsyncWebsocketConsumer):
    """
    Handles real-time streaming chat across all documents in a workspace.
    """
    async def connect(self):
        self.user = self.scope["user"]
        self.workspace_id = self.scope['url_route']['kwargs']['workspace_id']

        if not self.user.is_authenticated:
            await self.close()
            return

        self.room_group_name = f"workspace_{self.workspace_id}_{self.user.id}"

        has_permission = await self.check_workspace_permission()
        if not has_permission:
            await self.close()
            return

        if self.channel_layer is not None:
            await self.channel_layer.group_add(
                self.room_group_name,
                self.channel_name
            )

        await self.accept()
        logger.info(f"Workspace WS connected: {self.user.username} - Workspace {self.workspace_id}")

    async def disconnect(self, close_code):
        if hasattr(self, "room_group_name") and self.channel_layer is not None:
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )
        logger.info(f"Workspace WS disconnected: {self.user} - Code {close_code}")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            if message_type == 'chat_message':
                await self.handle_chat_message(data)
            elif message_type == 'typing':
                await self.handle_typing(data)
        except json.JSONDecodeError:
            await self.send_error("Invalid JSON format")
        except Exception as e:
            logger.error(f"Workspace chat receive error: {e}", exc_info=True)
            await self.send_error("An unexpected error occurred.")

    async def handle_chat_message(self, data):
        content = data.get('content', '').strip()
        if not content:
            return

        rate_limit_result = await self.is_rate_limited()
        if rate_limit_result.limited:
            await self.send_error(f"Rate limit exceeded. Please wait {rate_limit_result.retry_after} seconds.")
            return

        # Check Limits
        limit_reached = await self.check_chat_limit()
        if limit_reached:
            await self.send_error("You have reached your free plan chat limit of 500 messages. Please upgrade to Pro (₹199/mo) or Industry (₹1,999/mo) to continue.")
            return

        workspace = await self.get_workspace()
        if not workspace:
            await self.send_error("Workspace not found.")
            return

        has_any_docs = await self.has_any_documents(workspace)
        if not has_any_docs:
            await self.send_error("No documents found in this workspace. Please upload or add documents first.")
            return

        documents = await self.get_workspace_documents(workspace)
        if not documents:
            await self.send_error("The documents in this workspace are still being chunked and indexed. Please wait a moment until indexing finishes.")
            return

        session = await self.get_or_create_workspace_session(workspace)
        user_msg = await self.save_user_message(session, content)

        await self.send(text_data=json.dumps({
            'type': 'user_message',
            'id': user_msg.id,
            'content': content,
            'timestamp': user_msg.created_at.isoformat()
        }))

        chat_history = await self.get_chat_history(session)
        await self.process_workspace_ai_response(workspace, documents, session, content, chat_history)

    async def process_workspace_ai_response(self, workspace, documents, session, user_message, chat_history):
        import time
        start_time = time.time()
        try:
            is_limit_reached = await self.check_token_balance_status()
            if is_limit_reached:
                await self.send_error("Token limit reached. Please upgrade to continue.")
                return

            active_model = "DeepSeek v4 Flash" if self.user.is_premium else "Gemini 2.5 Flash"
            await self.send(text_data=json.dumps({
                'type': 'ai_thinking',
                'model': active_model
            }))

            from .utils.gemini_chat import get_workspace_response
            full_response = ""
            total_tokens_used = [0]

            async def _record_usage(total_tokens):
                total_tokens_used[0] = total_tokens
                usage = await self.deduct_user_tokens(total_tokens)
                await self.send(text_data=json.dumps({
                    'type': 'usage_update',
                    'usage': usage
                }))

            async def _progress(stage_text):
                await self.send(text_data=json.dumps({
                    'type': 'ai_progress',
                    'stage': stage_text
                }))

            async for chunk in get_workspace_response(
                user_message,
                documents,
                chat_history,
                progress_callback=_progress,
                usage_callback=_record_usage,
                user=self.user,
            ):
                if isinstance(chunk, dict) and chunk.get("type") == "thinking":
                    th_text = chunk.get("chunk", "")
                    if th_text:
                        await self.send(text_data=json.dumps({
                            'type': 'ai_thinking_chunk',
                            'content': th_text
                        }))
                else:
                    text_chunk = chunk.get("chunk", "") if isinstance(chunk, dict) else chunk
                    if text_chunk:
                        full_response += text_chunk
                        await self.send(text_data=json.dumps({
                            'type': 'ai_stream_chunk',
                            'content': text_chunk
                        }))

            latency_ms = int((time.time() - start_time) * 1000)
            ai_msg = await self.save_ai_message(
                session,
                full_response,
                tokens_used=total_tokens_used[0] if total_tokens_used[0] > 0 else None,
                latency_ms=latency_ms
            )

            await self.send(text_data=json.dumps({
                'type': 'ai_stream_end',
                'content': full_response,
                'id': ai_msg.id,
                'timestamp': ai_msg.created_at.isoformat(),
                'latency_ms': latency_ms,
            }))

        except Exception as e:
            logger.error(f"Workspace RAG Error: {e}", exc_info=True)
            await self.send_error("An error occurred while generating the multi-document answer.")

    async def handle_typing(self, data):
        if self.channel_layer:
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'typing_indicator',
                    'user': self.user.username
                }
            )

    async def typing_indicator(self, event):
        if event['user'] != self.user.username:
            await self.send(text_data=json.dumps({
                'type': 'user_typing',
                'user': event['user']
            }))

    async def send_error(self, message):
        await self.send(text_data=json.dumps({'type': 'error', 'message': message}))

    @database_sync_to_async
    def check_workspace_permission(self):
        from .models import Workspace
        try:
            Workspace.objects.get(id=self.workspace_id, owner=self.user)
            return True
        except Workspace.DoesNotExist:
            return False

    @database_sync_to_async
    def is_rate_limited(self):
        from .utils.rate_limit import check_user_rate_limit
        limit = 60 if self.user.is_premium else 10
        return check_user_rate_limit(self.user, "workspace_ws", limit, 60)

    @database_sync_to_async
    def check_token_balance_status(self):
        from payments.models import Subscription
        try:
            sub = Subscription.objects.get(user=self.user)
            if sub.is_token_metered:
                return sub.tokens_allocated <= 0
        except Subscription.DoesNotExist:
            pass
        return False

    @database_sync_to_async
    def deduct_user_tokens(self, total_tokens):
        from payments.services import deduct_tokens, get_usage_snapshot
        deduct_tokens(self.user, total_tokens)
        return get_usage_snapshot(self.user)

    @database_sync_to_async
    def get_workspace(self):
        from .models import Workspace
        try:
            return Workspace.objects.get(id=self.workspace_id, owner=self.user)
        except Workspace.DoesNotExist:
            return None

    @database_sync_to_async
    def has_any_documents(self, workspace):
        return workspace.documents.exists()

    @database_sync_to_async
    def get_workspace_documents(self, workspace):
        return list(workspace.documents.filter(is_processed=True))

    @database_sync_to_async
    def get_or_create_workspace_session(self, workspace):
        session, _ = ChatSession.objects.get_or_create(
            workspace=workspace,
            user=self.user,
            defaults={'title': f'Chat: {workspace.name}'}
        )
        return session

    @database_sync_to_async
    def save_user_message(self, session, content):
        return ChatMessage.objects.create(session=session, role="user", content=content)

    @database_sync_to_async
    def save_ai_message(self, session, content, tokens_used=None, latency_ms=None):
        return ChatMessage.objects.create(
            session=session,
            role="assistant",
            content=content,
            tokens_used=tokens_used,
            latency_ms=latency_ms
        )

    @database_sync_to_async
    def get_chat_history(self, session, limit=20):
        # Rolling window of the last 20 messages prevents latency bloat and context overflow
        messages = list(
            ChatMessage.objects.filter(session=session)
            .order_by('-created_at')[:limit]
        )
        messages.reverse()
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    @database_sync_to_async
    def check_chat_limit(self):
        # Pro and Industry subscribers are metered by token quotas, bypassing global message table scans
        if getattr(self.user, "is_premium", False):
            return False
        user_msg_count = ChatMessage.objects.filter(
            session__user=self.user, 
            role='user'
        ).count()
        return user_msg_count >= 500