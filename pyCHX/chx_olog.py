"""CHX Olog helpers with lazy client construction."""

from __future__ import annotations

from functools import lru_cache
from shutil import copyfile

from pyCHX.config import DEFAULT_OLOG_URL, get_olog_url

OLOG_URL = DEFAULT_OLOG_URL


def _missing_dependency_stub(symbol: str):
    message = f"{symbol} requires the optional 'pyOlog' dependency. " "Install pyCHX with the 'facility' extra."

    class MissingOptionalDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(message)

    MissingOptionalDependency.__name__ = symbol
    MissingOptionalDependency.__qualname__ = symbol
    MissingOptionalDependency.__doc__ = message
    return MissingOptionalDependency


try:
    from pyOlog import Attachment, LogEntry, SimpleOlogClient
    from pyOlog.OlogDataTypes import Logbook
except ImportError:
    Attachment = _missing_dependency_stub("Attachment")
    LogEntry = _missing_dependency_stub("LogEntry")
    Logbook = _missing_dependency_stub("Logbook")
    SimpleOlogClient = _missing_dependency_stub("SimpleOlogClient")


@lru_cache(maxsize=1)
def get_olog_client(url: str | None = None):
    """Construct and cache the Olog client on first use."""
    if url is None:
        url = get_olog_url()
    return SimpleOlogClient(url=url)


class _LazyOlogClient:
    """Compatibility proxy for code that imported ``olog_client``."""

    def __getattr__(self, name):
        return getattr(get_olog_client(), name)

    def __repr__(self):
        return "<OlogClient (lazy)>"


olog_client = _LazyOlogClient()


def create_olog_entry(text, logbooks="Data Acquisition"):
    """Create an entry in the CHX Olog."""
    return get_olog_client().log(text, logbooks=logbooks)


def update_olog_uid_with_file(uid, text, filename, append_name=""):
    """Attach text and a file to the CHX Olog entry containing *uid*."""
    client = get_olog_client()
    try:
        with open(filename, "rb") as stream:
            attachments = [Attachment(stream)]
            update_olog_uid(client, uid=uid, text=text, attachments=attachments)
    except Exception:
        new_name = f"{filename[:-4]}_{append_name}.pdf"
        copyfile(filename, new_name)
        print(f"Append {append_name} to the filename.")
        with open(new_name, "rb") as stream:
            attachments = [Attachment(stream)]
            update_olog_uid(client, uid=uid, text=text, attachments=attachments)


def update_olog_logid_with_file(logid, text, filename=None, verbose=False):
    """Attach text and optionally a file to an Olog entry."""
    try:
        if filename is None:
            update_olog_id(
                get_olog_client(),
                logid=logid,
                text=text,
                attachments=None,
                verbose=verbose,
            )
        else:
            with open(filename, "rb") as stream:
                update_olog_id(
                    get_olog_client(),
                    logid=logid,
                    text=text,
                    attachments=[Attachment(stream)],
                    verbose=verbose,
                )
    except Exception:
        pass


def update_olog_id(olog_client=None, logid=None, text=None, attachments=None, verbose=True):
    """Append text and attachments to an Olog entry selected by ID."""
    if olog_client is None:
        olog_client = get_olog_client()
    client = olog_client.session
    url = client._url
    old_text = olog_client.find(id=logid)[0]["text"]
    update = LogEntry(
        text=f"{old_text}\n{text}",
        attachments=attachments,
        logbooks=[Logbook(name="Operations", owner=None, active=True)],
    )
    client.updateLog(logid, update)
    if verbose:
        print(f"The url={url} was successfully updated with {text} and with the attachments")
    return old_text


def update_olog_uid(olog_client=None, uid=None, text=None, attachments=None):
    """Append text and attachments to the Olog entry containing *uid*."""
    if olog_client is None:
        olog_client = get_olog_client()
    logid = olog_client.find(search=f"*{uid}*")[-1]["id"]
    return update_olog_id(olog_client, logid, text, attachments)


__all__ = [
    "Attachment",
    "LogEntry",
    "create_olog_entry",
    "get_olog_client",
    "olog_client",
    "update_olog_id",
    "update_olog_logid_with_file",
    "update_olog_uid",
    "update_olog_uid_with_file",
]
