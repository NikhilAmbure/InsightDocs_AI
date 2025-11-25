
import asyncio
import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import Document, ChatSession, ChatMessage
from .utils.gemini_chat import get_gemini_response
from .utils.storage import prepare_local_document

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

        # Check if user has permission to access document
        has_permission = await self.check_document_permission()
        
        if not has_permission:
            await self.close()
            return

        # Join room group (if channel_layer is configured)
        if self.channel_layer is not None:
            try:
                await self.channel_layer.group_add(
                    self.room_group_name,
                    self.channel_name
                )
            except Exception as e:
                logger.error(f"Error adding to group: {str(e)}")
        
        await self.accept()
        logger.info(f"WebSocket connected: {self.user.username} - Doc {self.document_id}")

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        # Leave room group (if channel_layer is configured)
        if self.channel_layer is not None:
            try:
                await self.channel_layer.group_discard(
                    self.room_group_name,
                    self.channel_name
                )
            except Exception as e:
                logger.error(f"Error removing from group: {str(e)}")
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
            else:
                await self.send_error("Unknown message type")
                
        except json.JSONDecodeError:
            await self.send_error("Invalid JSON format")
        except Exception as e:
            logger.error(f"Error in receive: {str(e)}", exc_info=True)
            await self.send_error(f"Server error: {str(e)}")

    async def handle_chat_message(self, data):
        """Handle incoming chat message"""
        content = data.get('content', '').strip()
        
        if not content:
            await self.send_error("Message cannot be empty")
            return

        # Get document and session
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

        # Process with Gemini (offload to thread pool)
        await self.process_ai_response(document, session, content, chat_history)

    async def handle_typing(self, data):
        """Broadcast typing indicator"""
        if self.channel_layer is not None:
            try:
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'typing_indicator',
                        'user': self.user.username
                    }
                )
            except Exception as e:
                logger.error(f"Error sending typing indicator: {str(e)}")

    async def process_ai_response(self, document, session, user_message, chat_history):
        """Process message through Gemini AI"""
        try:
            # Notify client that AI is processing
            await self.send(text_data=json.dumps({
                'type': 'ai_thinking',
                'status': 'processing'
            }))

            # Ensure we have a local copy of the document for Gemini ingestion
            local_path, cleanup = await asyncio.to_thread(prepare_local_document, document)

            try:
                ai_response = await self.get_gemini_response_async(
                    user_message,
                    local_path,
                    chat_history
                )
            finally:
                await asyncio.to_thread(cleanup)

            # Save AI message
            ai_msg = await self.save_ai_message(session, ai_response)

            # Send AI response to client
            await self.send(text_data=json.dumps({
                'type': 'ai_message',
                'id': ai_msg.id,
                'content': ai_response,
                'timestamp': ai_msg.created_at.isoformat()
            }))

        except Exception as e:
            logger.error(f"Error processing AI response: {str(e)}", exc_info=True)
            await self.send_error(f"AI Error: {str(e)}")

    async def send_error(self, message):
        """Send error message to client"""
        await self.send(text_data=json.dumps({
            'type': 'error',
            'message': message
        }))

    # Typing indicator handler
    async def typing_indicator(self, event):
        """Send typing indicator to WebSocket"""
        # Only send if it's not from the same user
        if event['user'] != self.user.username:
            await self.send(text_data=json.dumps({
                'type': 'user_typing',
                'user': event['user']
            }))

    # Database operations
    @database_sync_to_async
    def check_document_permission(self):
        """Check if user has permission to access document"""
        try:
            Document.objects.get(id=self.document_id, owner=self.user)
            return True
        except Document.DoesNotExist:
            return False

    @database_sync_to_async
    def get_document(self):
        """Get document from database"""
        try:
            return Document.objects.get(id=self.document_id, owner=self.user)
        except Document.DoesNotExist:
            return None

    @database_sync_to_async
    def get_or_create_session(self, document):
        """Get or create chat session"""
        session, _ = ChatSession.objects.get_or_create(
            document=document,
            user=self.user
        )
        return session

    @database_sync_to_async
    def save_user_message(self, session, content):
        """Save user message to database"""
        return ChatMessage.objects.create(
            session=session,
            role="user",
            content=content
        )

    @database_sync_to_async
    def save_ai_message(self, session, content):
        """Save AI message to database"""
        return ChatMessage.objects.create(
            session=session,
            role="assistant",
            content=content
        )

    @database_sync_to_async
    def get_chat_history(self, session):
        """Get chat history for context"""
        messages = ChatMessage.objects.filter(session=session).order_by('created_at')
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    @database_sync_to_async
    def get_gemini_response_async(self, user_message, file_path, chat_history):
        """Async wrapper for Gemini response"""
        return get_gemini_response(user_message, file_path, chat_history)