import asyncio
import json
import logging

from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
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
        if self.channel_layer is not None:
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )
        logger.info(f"WebSocket disconnected: {self.user.username} - Code {close_code}")

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
        
        # Check Limits
        limit_reached = await self.check_chat_limit()
        if limit_reached:
            await self.send_error("You have reached your chat message limit.")
            return

        # Get Document & Session
        document = await self.get_document()
        if not document:
            await self.send_error("Document not found")
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

        # Process with Gemini
        await self.process_ai_response(document, session, content, chat_history)

    async def process_ai_response(self, document, session, user_message, chat_history):
        """Process message through Gemini AI using RAG (Streaming)"""
        try:
            # 1. Notify client that AI is thinking
            await self.send(text_data=json.dumps({'type': 'ai_thinking'}))

            full_response = ""

            # 2. Stream chunks to the client as we receive them
            async for chunk in get_gemini_response(user_message, document, chat_history):
                full_response += chunk
                await self.send(text_data=json.dumps({
                    'type': 'ai_stream_chunk',
                    'content': chunk
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
    def get_chat_history(self, session):
        messages = ChatMessage.objects.filter(session=session).order_by('created_at')
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    @database_sync_to_async
    def check_chat_limit(self):
        user_msg_count = ChatMessage.objects.filter(
            session__user=self.user, 
            role='user'
        ).count()
        limit = 5000 if self.user.is_premium else 500
        return user_msg_count >= limit