import asyncio
from typing import List, Dict, Any
from fastapi import WebSocket
from vaxguard.core.logger import get_logger

logger = get_logger("WebSocketManager")


class ConnectionManager:
    """
    Thread/async-safe manager for live WebSocket connections.
    Streams real-time threat detection alerts, scan events, and auto-immunity notifications.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total active: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Remaining active: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcasts event payload to all connected clients."""
        async with self._lock:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.debug(f"Failed to send to client, marking for removal: {e}")
                    disconnected.append(connection)

            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)

    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]):
        """Sends message to a single specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
