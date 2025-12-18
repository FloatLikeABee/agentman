import asyncio
import time
import websockets
import json
import logging


class WebSocketClient:
    def __init__(self, uri, on_message=None, on_error=None):
        self.uri = uri
        self.websocket = None
        self.on_message = on_message or self.default_on_message
        self.on_error = on_error or self.default_on_error
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.uri)
            self.logger.info(f"Connected to {self.uri}")
            return self
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            raise

    async def send(self, message):
        if not self.websocket:
            raise ConnectionError("Not connected to WebSocket")
        
        try:
            if isinstance(message, dict):
                message = json.dumps(message)
            
            await self.websocket.send(message)
            self.logger.info(f"Sent: {message}")
        except Exception as e:
            self.logger.error(f"Send error: {e}")
            await self.close()

    async def receive(self):
        try:
            message = await self.websocket.recv()
            return message
        except websockets.ConnectionClosed:
            self.logger.warning("Connection closed")
            return None

    async def listen(self):
        try:
            while True:
                message = await self.receive()
                if message:
                    self.on_message(message)
        except Exception as e:
            self.on_error(e)
        finally:
            await self.close()

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            self.logger.info("WebSocket connection closed")

    def default_on_message(self, message):
        print(f"Received: {message}")

    def default_on_error(self, error):
        print(f"Error occurred: {error}")

# Example usage
async def main():
    def on_message(msg):
        print(f"Custom message handler: {msg}")

    def on_error(err):
        print(f"Custom error handler: {err}")

    client = await WebSocketClient(
        "ws://127.0.0.1:8196/message",
        on_message=on_message,
        on_error=on_error
    ).connect()

    time_sent = int(time.time() * 1000)

    # content_struct = {
    #             "dataItem": "message",
    #             "fromIdentity": "gee",
    #             "toIdentity": "bee",
    #             "fromUsername": "floatingGee",
    #             "messageType": "send",
    #             "body": json.dumps({
    #                 "serial": "00000000000000",
    #                 "timestamp": time_sent,
    #                 "dateTime": "",
    #                 "member": {
    #                     "id": 0,
    #                     "background": "",
    #                     "icon": "",
    #                     "username": ""
    #                 },
    #                 "type": "text",
    #                 "text": "Test Text Number One",
    #                 "image": {},
    #                 "voice": {},
    #                 "video": {},
    #                 "pointers": []
    #             }),
    #             "timestamp": time_sent
    #         }
    
    content_struct = {
                "dataItem": "message",
                "fromIdentity": "gee",
                "toIdentity": "bee",
                "fromUsername": "floatingBee",
                "messageType": "get",
                "body": json.dumps({
                    "serial": "00000000000000",
                    "timestamp": time_sent,
                    "dateTime": "",
                    "member": {
                        "id": 0,
                        "background": "",
                        "icon": "",
                        "username": ""
                    },
                    "type": "text",
                    "text": "",
                    "image": {},
                    "voice": {},
                    "video": {},
                    "pointers": []
                }),
                "timestamp": time_sent
            }

    # Send a message
    await client.send(
        {
            "operation": "communication",
            "content": json.dumps(content_struct)
        }
    )

    # Start listening
    await client.listen()

# Run the async function
asyncio.run(main())
