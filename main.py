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

ENTRIES_PER_SEARCH_PAGE = 24
SEARCH_TERMS = ["guide"]


async def download_url(url: str, session: ClientSession, tg: TaskGroup, save=True):
    url_found = 0
    if not url.startswith("https://www.seriouseats.com/"):
        print("Wrong URL:", url)
        return url_found
    try:
        async with await aiofile.async_open("results/" + url.split("/")[-1] + ".html", mode="xb+") if save else nullcontext() as file, session.get(
                url) as request:
            if request.status != 200:
                print(request.status, await request.text())
                return url_found

            text = await request.read()
            document = parse(StringIO(str(text)))
            for u in (i.get("href") for i in
                        document.xpath('//a[contains(concat(" ", normalize-space(@class), " "), " card ")]')):
                tg.start_soon(download_url, u, session, tg)
                url_found += 1
            if save:
                print(url)
                await file.write(text)
    except FileExistsError:
        pass
    except aiohttp.InvalidURL:
        print("Invalid URL:", url)
        pass
    except Exception as e:
        print("Exception", e)

    return url_found


async def start():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1),
                                     timeout=ClientTimeout(sock_read=5, sock_connect=5)) as session:
        async with create_task_group() as tg:
            for seach_term in SEARCH_TERMS:
                offset = 0
                while await download_url(f'https://www.seriouseats.com/search?q={seach_term}&offset={offset}', session, tg, save=False):
                    offset += ENTRIES_PER_SEARCH_PAGE


async def enrich():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1),
                                     timeout=ClientTimeout(sock_read=5, sock_connect=5)) as session:
        async with create_task_group() as tg:
            # Using pathlib.Path.iterdir would be undefined behaviour since we modified the directory while iterating.
            for path in os.listdir("results"):
                document = parse("results/" + path)
                try:
                    for url in (i.get("href") for i in
                                document.xpath('//a[contains(concat(" ", normalize-space(@class), " "), " card ")]')):
                        tg.start_soon(download_url, url, session, tg)
                except AssertionError as e:
                    os.remove("results/" + path)
                    print("Removing corrupted:", path)
                print(path)
            await anyio.sleep(0)  # yield that we don't queue up too much


if __name__ == "__main__":
    run(start)
    # run(enrich)
