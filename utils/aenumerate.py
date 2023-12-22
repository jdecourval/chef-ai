async def aenumerate(asequence, start=0):
    """Asynchronously enumerate an async iterator from a given start value"""
    # From https://stackoverflow.com/a/55930068
    n = start
    async for elem in asequence:
        yield n, elem
        n += 1
