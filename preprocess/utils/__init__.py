import zenodo_client

def zenodo_get(record: str,
               filename: str):
    zen = zenodo_client.Zenodo()
    path = zen.download_latest(record_id=record, path=filename)
    return path
