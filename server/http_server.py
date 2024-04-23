import asyncio
import logging
import os
import ssl
import time

import aiohttp
from aiohttp import web
import aiohttp_jinja2
import jinja2
import jwt

import credentials

ROLE_HOST = 1
ROLE_PARTICIPANT = 0
USER_NAME = 'Human Expert'
IMAGES_DIR = 'images'
TOKEN_VALID_HRS = 24

KEYS = 'keys'
CERTFILE = os.path.join(KEYS, 'fullchain.pem')
KEYFILE = os.path.join(KEYS, 'privkey.pem')

logger = logging.getLogger(__name__)


@aiohttp_jinja2.template('zoom.html')
async def zoom(request):
    jwt_token = gen_jwt_token(credentials.CLIENT_ID, credentials.CLIENT_SECRET,
                              credentials.MEETING_NUMBER, ROLE_HOST)
    return {
        'meeting_number': credentials.MEETING_NUMBER,
        'user_name': USER_NAME,
        'user_email': credentials.USER_EMAIL,
        'password': credentials.MEETING_PASSWORD,
        'client_id': credentials.CLIENT_ID,
        'jwt_token': jwt_token
    }


async def favicon(request):
    return web.FileResponse('favicon.ico')


def gen_jwt_token(key, secret, meeting_number, role):
    iat = int(round(time.time())) - 30
    exp = iat + 3600 * TOKEN_VALID_HRS
    header = {'alg': 'HS256', 'typ': 'JWT'}
    payload = {
        'sdkKey': key,
        'appKey': key,
        'mn': meeting_number,
        'role': role,
        'iat': iat,
        'exp': exp,
        'tokenExp': exp
    }
    return jwt.encode(payload, secret, algorithm="HS256", headers=header)


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
