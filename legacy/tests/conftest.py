import os
from types import SimpleNamespace

import aiohttp.client_reqrep as client_reqrep


def pytest_sessionstart(session):
    os.environ["BM25"] = "bm25"
    _allow_aioresponses_without_stream_writer()


def _allow_aioresponses_without_stream_writer():
    """Let aioresponses 0.7.9 construct aiohttp 3.14 responses.

    ClientResponse now requires stream_writer and reads output_size when the
    request writer is already finished. Real sessions pass a writer. This
    default exists only for the mock, which still omits the argument.
    """
    original = client_reqrep.ClientResponse.__init__

    def init(
        self,
        method,
        url,
        *,
        writer,
        continue100,
        timer,
        request_info,
        traces,
        loop,
        session,
        stream_writer=None,
    ):
        if stream_writer is None:
            stream_writer = SimpleNamespace(output_size=0)
        return original(
            self,
            method,
            url,
            writer=writer,
            continue100=continue100,
            timer=timer,
            request_info=request_info,
            traces=traces,
            loop=loop,
            session=session,
            stream_writer=stream_writer,
        )

    client_reqrep.ClientResponse.__init__ = init
