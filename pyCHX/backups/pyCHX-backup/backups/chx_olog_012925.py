from pyOlog import Attachment, LogEntry, OlogClient, SimpleOlogClient
from pyOlog.OlogDataTypes import Logbook


def create_olog_entry(text, logbooks="Data Acquisition"):
    """
    Create a log entry to xf11id.

    Parameters
    ----------
    text : str
        the text string to add to the logbook
    logbooks : str, optional
        the name of the logbook to update

    Returns
    -------
    eid : the entry id returned from the Olog server
    """
    olog_client = SimpleOlogClient()
    eid = olog_client.log(text, logbooks=logbooks)
    return eid


def update_olog_uid_with_file(uid, text, filename, append_name=""):
    """
    Attach text and file (with filename) to CHX olog with entry defined by uid.

    Parameters
    ----------
    uid : str
        string of unique id
    text : str
        string to put into olog book
    filename : str
        file name
    append_name : str
        first try to attach olog with the file, if there is already a same file
        in attached file, copy the file with different filename (append
        append_name), and then attach to olog
    """
    atch = [Attachment(open(filename, "rb"))]

    try:
        update_olog_uid(uid=uid, text=text, attachments=atch)
    except Exception:
        from shutil import copyfile

        npname = f"{filename[:-4]}_{append_name}.pdf"
        copyfile(filename, npname)
        atch = [Attachment(open(npname, "rb"))]
        print(f"Append {append_name} to the filename.")
        update_olog_uid(uid=uid, text=text, attachments=atch)


def update_olog_logid_with_file(logid, text, filename=None, verbose=False):
    """
    Attach text and file (with filename) to CHX olog with entry defined by
    logid.

    Parameters
    ----------
    logid : str
        the log entry id
    text : str
        string to put into olog book
    filename : str
        file name
    """
    if filename is not None:
        atch = [Attachment(open(filename, "rb"))]
    else:
        atch = None
    try:
        update_olog_id(logid=logid, text=text, attachments=atch, verbose=verbose)
    except Exception:
        pass


def update_olog_id(logid, text, attachments, verbose=True):
    """
    Update olog book logid entry with text and attachments files.

    Parameters
    ----------
    logid : integer
        the log entry id
    text : str
        the text to update, will add this text to the old text
    attachments : ???
        add new attachment files

    Example
    -------
    filename1 = ('/XF11ID/analysis/2016_2/yuzhang/Results/August/af8f66/'
                 'Report_uid=af8f66.pdf')
    atch = [Attachment(open(filename1, 'rb'))]

    update_olog_id(logid=29327, text='add_test_atch', attachmenents=atch)
    """
    olog_client = SimpleOlogClient()
    client = OlogClient()
    url = client._url

    old_text = olog_client.find(id=logid)[0]["text"]
    upd = LogEntry(
        text=f"{old_text}\n{text}",
        attachments=attachments,
        logbooks=[Logbook(name="Operations", owner=None, active=True)],
    )
    client.updateLog(logid, upd)
    if verbose:
        print(f"The url={url} was successfully updated with {text} and with " f"the attachments")


def update_olog_uid(uid, text, attachments):
    """
    Update olog book logid entry cotaining uid string with text and attachments
    files.

    Parameters
    ----------
    uid: str
        the uid of a scan or a specficial string (only gives one log entry)
    text: str
        the text to update, will add this text to the old text
    attachments: ???
        add new attachment files

    Example
    -------
    filename1 = ('/XF11ID/analysis/2016_2/yuzhang/Results/August/af8f66/'
                 'Report_uid=af8f66.pdf')
    atch = [Attachment(open(filename1, 'rb'))]
    update_olog_uid(uid='af8f66', text='Add xpcs pdf report', attachments=atch)
    """
    olog_client = SimpleOlogClient()

    logid = olog_client.find(search=f"*{uid}*")[0]["id"]
    update_olog_id(logid, text, attachments)
