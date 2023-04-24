from channels.routing import ProtocolTypeRouter

from django.urls import path

from . import consumers

websocket_urlpatterns = [
    path("predict", consumers.PredictionConsumer.as_asgi()),
]