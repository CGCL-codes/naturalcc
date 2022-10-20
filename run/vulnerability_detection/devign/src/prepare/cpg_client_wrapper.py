import os
import requests
import time

from cpgclient.CpgClient import CpgClient


class CPGClientWrapper(CpgClient):
    def __init__(self, address='127.0.0.1', port=8080):
        super(CPGClientWrapper, self).__init__(address, port)
        self.abs_path = os.path.dirname(os.path.abspath(os.getcwd()))
        self.query_script = f"cpg.runScript(\"{self.abs_path}/joern/graph-for-funcs.sc\")"

    def __call__(self, out_path):
        self.create_cpg(self.abs_path + out_path)
        return self.query(self.query_script)

    def _wait_until_cpg_is_created(self):
        while not self.is_cpg_loaded():
            time.sleep(1)

    def _poll_for_query_result(self):
        while True:
            response = requests.get("{}/v1/query/{}".format(self.handlerAndUrl, self.currentQueryId))
            json_body = response.json()
            if json_body["ready"]:
                return json_body["result"] or json_body["error"]
            time.sleep(.1)