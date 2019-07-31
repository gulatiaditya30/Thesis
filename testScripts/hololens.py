import json
import logging
from threading import Thread

from websocket_server import WebsocketServer

from queue import Queue

START_RIVET_POS = (-0.426, 0.468, 0.105)

logger = logging.getLogger(__name__)

_msgqueue = Queue()


def queue_message(quality, rivet=(0, 0, 0, 0, 0, 0)):
    try:
        q = quality.name
    except AttributeError:
        q = quality
    msg = {'messageType': 'rivetCoordinates', 'quality': q}
    msg.update(zip('x y z a b c'.split(), rivet))
    j = json.dumps(msg)
    print(j)
    _msgqueue.put(j)


def create_websocket_server(host='localhost', port=8000):
    server = WebsocketServer(port, host)
    server.set_fn_new_client(send_messages)
    return server


def start_websocket_server(host='localhost', port=8000):
    server = create_websocket_server(host, port)
    server_thread = Thread(target=server.run_forever, args=())
    server_thread.setDaemon(True)
    server_thread.start()


def send_messages(client, server):
    print('Connected to ', client['address'][0])
    while True:
        msg = _msgqueue.get()
        server.send_message_to_all(msg)


if __name__ == '__main__':
    start_websocket_server(host='')
