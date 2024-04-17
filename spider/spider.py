import logging
import os
from contextlib import nullcontext
from io import StringIO

import aiofile
import aiohttp
import anyio
from aiohttp import ClientSession, ClientTimeout
from anyio import run, create_task_group
from anyio.abc import TaskGroup
from lxml.html import parse
from tqdm import tqdm

ENTRIES_PER_SEARCH_PAGE = 24
SEARCH_TERMS = ["recipe", "food", "salt", "fork", "knife", "guide", "best", "why", "what", "when", "how", "mg", "oz"]
HTTP_CONNECTOR = aiohttp.TCPConnector(limit=1)  # Cheap way to avoid being rate limited.
TIMEOUT = ClientTimeout(sock_read=1, sock_connect=1)
BASE_URL = os.environ["SPIDER_BASE_URL"]

_logger = logging.getLogger(__name__)


def find_related_urls(text: bytes, session: ClientSession, tg: TaskGroup):
    urls_found = 0
    document = parse(StringIO(text.decode()))
    for urls_found, url in enumerate(
            i.get("href") for i in document.xpath('//a[contains(concat(" ", normalize-space(@class), " "), " card ")]')
    ):
        tg.start_soon(download_url, url, session, tg)
    return urls_found


async def download_url(url: str, session: ClientSession, tg: TaskGroup, save=True):
    urls_found = 0
    if not url.startswith(f"{BASE_URL}/"):
        _logger.warning(f"Found an external link: {url}")
        return urls_found
    try:
        # With mode="xb+", if the resulting file exists already, we won't download again.
        # This syntax behaves like two nested 'with', this means session.get is not called if opening the file fails.
        async with (await aiofile.async_open("results/" + url.split("/")[-1] + ".html", mode="xb+") if save
                    else nullcontext() as file,
                    session.get(url) as request):
            if request.status != 200:
                _logger.error(f"{request.status} status with url: {url}")
                return urls_found

            text = await request.read()
            urls_found = find_related_urls(text, session, tg)
            if save:
                _logger.info(f"Saving new document: {url.split('/')[-1]}")
                await file.write(text)
    except FileExistsError:
        _logger.debug(f"URL already downloaded: {url}")  # mode="xb+" will cause this exception if the file exists.
    except aiohttp.InvalidURL:
        _logger.error(f"Invalid URL: {url}")
    except Exception as e:
        _logger.error(f"Unexpected error with url: {url}: {e}")

    return urls_found


async def start():
    async with aiohttp.ClientSession(connector=HTTP_CONNECTOR, timeout=TIMEOUT) as session:
        async with create_task_group() as tg:
            for seach_term in SEARCH_TERMS:
                offset = 0
                while await download_url(f'{BASE_URL}/search?q={seach_term}&offset={offset}',
                                         session,
                                         tg,
                                         save=False):
                    offset += ENTRIES_PER_SEARCH_PAGE


async def enrich():
    """
    Enrich the dataset by going through the downloaded files to find more URL.
    This shouldn't be necessary since `start` also enriches all the URLs.
    """
    async with aiohttp.ClientSession(connector=HTTP_CONNECTOR, timeout=TIMEOUT) as session:
        async with create_task_group() as tg:
            # Using pathlib.Path.iterdir would be undefined behaviour since we modify the directory while iterating.
            for path in tqdm(os.listdir("results")):
                document = parse("results/" + path)
                try:
                    find_related_urls(document, session, tg)
                except AssertionError as e:
                    os.remove("results/" + path)
                    _logger.exception(f"Found a corrupted file, removing it: {path}", e)
            await anyio.sleep(0)  # yield that we don't queue up too much


if __name__ == "__main__":
    run(start)
    # run(enrich)
