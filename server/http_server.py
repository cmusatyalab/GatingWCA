import asyncio
import credentials
import logging
import os
import ssl
import aiohttp
from aiohttp import web
import aiohttp_jinja2
import jinja2


ROLE = '1'
USER_NAME = 'Human Expert'
IMAGES_DIR = 'images'
TOKEN_VALID_HRS = 24


KEYS = 'keys'
CERTFILE = os.path.join(KEYS, 'fullchain.pem')
KEYFILE = os.path.join(KEYS, 'privkey.pem')


logger = logging.getLogger(__name__)


@aiohttp_jinja2.template('zoom.html')
async def zoom(request):
    return {
        'meeting_number': credentials.MEETING_NUMBER,
        'user_name': USER_NAME,
        'user_email': credentials.USER_EMAIL,
        'role': ROLE,
        'password': credentials.MEETING_PASSWORD,
        'client_id': credentials.CLIENT_ID,
        'client_secret': credentials.CLIENT_SECRET,
        'token_valid_hrs': TOKEN_VALID_HRS
    }


async def favicon(request):
    return web.FileResponse('favicon.ico')


class _ServerState:
    def __init__(self):
        self._step = None
        self._user_connected = False

    def create_websocket_handler(self, conn):
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            if self._user_connected:
                logger.info('ignoring additional websocket connection')
                return ws

            self._user_connected = True
            logger.info('websocket connection opened')

            def engine_reader():
                from_engine = conn.recv()
                asyncio.ensure_future(ws.send_json(from_engine))
                if from_engine.get('zoom_action') == 'start':
                    self._step = from_engine.get('step')
                elif from_engine.get('zoom_action') == 'stop':
                    logger.info('Zoom call ended on step: %s', self._step)
                    conn.send({
                        'step': self._step
                    })

            asyncio.get_event_loop().add_reader(
                conn.fileno(), engine_reader)

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._step = msg.json().get('step')
                    logger.info('Web user set step: %s', self._step)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error('ws connection closed with exception %s',
                                 ws.exception())

            self._user_connected = False
            logger.info('websocket connection closed')

            return ws
        return websocket_handler

    def get_user_connected(self):
        return self._user_connected


def start_http_server(conn, step_names):
    app = web.Application()
    aiohttp_jinja2.setup(
        app, loader=jinja2.FileSystemLoader('templates'))

    server_state = _ServerState()
    websocket_handler = server_state.create_websocket_handler(conn)

    @aiohttp_jinja2.template('index.html')
    async def index(request):
        return {
            'duplicate_connection': server_state.get_user_connected(),
            'step_names': step_names
        }

    app.add_routes([
        web.get('/', index),
        web.get('/zoom', zoom),
        web.get('/favicon.ico', favicon),
        web.static('/static', 'static'),
        web.static('/{}'.format(IMAGES_DIR), IMAGES_DIR),
        web.get('/ws', websocket_handler),
    ])

    context = ssl.SSLContext()
    context.load_cert_chain(CERTFILE, KEYFILE)
    logger.info("Starting HTTP Server")
    web.run_app(app, ssl_context=context)
