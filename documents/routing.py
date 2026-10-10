from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/chat/(?P<document_id>\w+)/$', consumers.ChatConsumer.as_asgi()),
    re_path(r'ws/workspace/(?P<workspace_id>\w+)/$', consumers.WorkspaceChatConsumer.as_asgi()),
    re_path(r'ws/editor/(?P<document_id>\w+)/$', consumers.EditorConsumer.as_asgi()),
]