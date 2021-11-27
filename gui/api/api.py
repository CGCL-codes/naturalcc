import json
import time
from ncc.cli.predictor import cli_main

from flask import Flask, request

app = Flask(__name__)


# CORS(app, resources=r'/*')
# app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/hi')
def hi():
    return 'hi~'


# api_prefix = '/api/'

# headers = {
#     'Cache-Control' : 'no-cache, no-store, must-revalidate',
#     'Pragma' : 'no-cache' ,
#     'Expires': '' ,
#     'Access-Control-Allow-Origin' : 'http://127.0.0.1:3001',
#     'Access-Control-Allow-Origin' : '*',
#     'Access-Control-Allow-Methods': 'GET, POST, PATCH, PUT, DELETE, OPTIONS',
#     'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token'
# }

@app.route('/api/time', methods=['POST'])
def get_current_time():
    # return {'time': time.time()}
    inputs = request.get_data(as_text=True)
    output = {}
    output["top_tokens"] = ['aaaaaaa', 'bbbbbbb', 'cccccc', 'ddddd']
    output["top_indices"] = [1, 3, 5, 6]
    output["probabilities"] = [65, 44, 12, 4]
    output["time"] = time.time()
    return json.dumps(output)


@app.route('/api/predict', methods=['POST'])  # , methods=['POST']
# @cross_origin()
def predict():
    inputs = request.get_data(as_text=True)
    # input = inputs["sentence"]
    # tt = request.args.get('tt')
    # # re = {
    # # 'code': 0,
    # # 'data':'xxxxdata',
    # # 'message': "这是测试呀"
    # # }

    # model_input = ujson.loads(inputs)["sentence"]
    # model_path = '~/.ncc/demo/completion/seqrnn/py150.pt'
    #
    # topk_info = cli_main(os.path.expanduser(model_path), input=model_input)
    # top_tokens, probabilities = list(zip(*topk_info))
    # output = {
    #     'top_tokens': [token + ' ' for token in top_tokens],
    #     'probabilities': probabilities,
    # }

    # invoke our model API and obtain the output
    output = {}
    print(inputs)

    # if ujson.loads(inputs)["sentence"].strip().endswith('send('):
    #     output["top_tokens"] = ['request', 'self', 'response', 'url']
    #     output["top_indices"] = [27, 4, 37, 42]
    #     output["probabilities"] = [0.7793, 0.1738, 0.0322, 0.0016]
    # elif ujson.loads(inputs)["sentence"].strip().endswith('request'):
    #     output["top_tokens"] = ['header_parameters', 'response', 'self', 'request']
    #     output["top_indices"] = [37, 1977, 4, 27, ]
    #     output["probabilities"] = [0.6704, 0.2045, 0.0390, 0.0329, ]
    # elif ujson.loads(inputs)["sentence"].strip().endswith('header_parameters'):
    #     output["top_tokens"] = ['body_content', 'response', 'Content-Type', 'operation_config']
    #     output["top_indices"] = [20592, 37, 1034, 4519, ]
    #     output["probabilities"] = [0.9858, 0.0046, 0.0035, 0.0024, ]
    # elif ujson.loads(inputs)["sentence"].strip().endswith('body_content'):
    #     output["top_tokens"] = ['operation_config', 'self', 'response', 'Content-Length']
    #     output["top_indices"] = [4519, 4, 37, 2849, ]
    #     output["probabilities"] = [1, 0.0001, 3.0696e-05, 1.2577e-05, ]
    # else:
    #     output["top_tokens"] = ['response', 'request', 'self', 'body']
    #     output["top_indices"] = [37, 27, 4, 129]
    #     output["probabilities"] = [0.9868, 0.0025, 0.0021, 0.0017]

    inputs = json.loads(inputs)
    model_path = f"~/ncc_data/raw_py150/completion/data-mmap/{str.lower(inputs['model'])}/checkpoints/checkpoint_best.pt"
    input = inputs['sentence']
    output = cli_main(model_path, input)

    output["top_tokens"] = [' ' + token for token in output["top_tokens"]]

    # rsp = flask.Response(json.dumps(output))
    # rsp.headers = headers
    # rsp.headers['Cache-Control'] = 'no-cache'
    # return rsp

    return json.dumps(output)


@app.route('/api/retrieve', methods=['POST'])  # , methods=['POST']
# @cross_origin()
def retrieve():
    inputs = request.get_data(as_text=True)
    # input = inputs["sentence"]
    # tt = request.args.get('tt')
    # # re = {
    # # 'code': 0,
    # # 'data':'xxxxdata',
    # # 'message': "这是测试呀"
    # # }

    # # invoke our model API and obtain the output
    # model_path = '~/.ncc/demo/retrieval/nbow/csn_ruby.pt'
    # model_input = ujson.loads(inputs)["utterance"]
    # raw_code = cli_main(os.path.expanduser(model_path), input=model_input)
    # raw_code = ujson.loads(raw_code)
    # output = {'predicted_sql_query': raw_code}

    output = {'predicted_sql_query': ''}

    inputs = json.loads(inputs)
    print(inputs)
    # if (inputs["model"] == "NBOW"):
    #     if inputs["utterance"].startswith("get_manifest should return an"):
    #         output["predicted_sql_query"] = "def download_layers(self, repo_name, digest=None, destination=None):\n" \
    #                                         "\tfrom sregistry.main.workers import ( Workers, download_task )\n" \
    #                                         "\tif not hasattr(self, 'manifests'):\n" \
    #                                         "\t\tself._get_manifests(repo_name, digest)\n" \
    #                                         '\tdigests = self._get_digests()\n' \
    #                                         '\tdestination = self._get_download_cache(destination)\n' \
    #                                         '\tworkers = Workers()\n' \
    #                                         "\ttasks = []\n" \
    #                                         '\tlayers = []\n' \
    #                                         '\tfor digest in digests:\n' \
    #                                         '\t\ttargz = "%s\/%s.tar.gz" % (destination, digest)\n' \
    #                                         '\t\tif not os.path.exists(targz):\n' \
    #                                         '\t\t\turl = "%s\/%s\/blobs\/%s" % (self.base, repo_name, digest)\n' \
    #                                         '\t\t\ttasks.append((url, self.headers, targz))\n' \
    #                                         '\t\tlayers.append(targz)\n' \
    #                                         '\tif len(tasks) > 0:\n' \
    #                                         '\t\tdownload_layers = workers.run(func=download_task,\n' \
    #                                         '\t\t\t\t\t\t\t\t\t\t\ttasks=tasks)\n' \
    #                                         '\tmetadata = self._create_metadata_tar(destination)\n' \
    #                                         '\tif metadata is not None:\n' \
    #                                         '\tlayers.append(metadata)\n' \
    #                                         '\treturn layers\n'
    #     elif inputs["utterance"].startswith("Add routes by an resource"):
    #         output["predicted_sql_query"] = "def get_url_args(url):\n" \
    #                                         '\turl_data = urllib.parse.urlparse(url)\n' \
    #                                         '\targ_dict = urllib.parse.parse_qs(url_data.query)\n' \
    #                                         '\treturn arg_dict\n',
    # elif (inputs["model"] == "BiRNN"):
    #     if inputs["utterance"].startswith("get_manifest should return an"):
    #         output["predicted_sql_query"] = "def top_group(\n" \
    #                                         "\t\tdf,\n" \
    #                                         "\t\taggregate_by: List[str],\n" \
    #                                         "\t\tvalue: str,\n" \
    #                                         "\t\tlimit: int,\n" \
    #                                         "\t\torder: str = 'asc',\n" \
    #                                         "\t\tfunction: str = 'sum',\n" \
    #                                         "\t\tgroup: Union[str, List[str]] = None\n" \
    #                                         "):\n" \
    #                                         "\tdf[column] = pd.to_datetime(df[column], format=format)\n" \
    #                                         "\treturn df\n"
    #     elif inputs["utterance"].startswith("Add routes by an resource"):
    #         output["predicted_sql_query"] = "def models(self):\n" \
    #                                         "\tapi_version = self._get_api_version(None)\n" \
    #                                         "\tif api_version == v7_0_VERSION:\n" \
    #                                         "\t\tfrom azure.keyvault.v7_0 import models as implModels\n" \
    #                                         "\telif api_version == v2016_10_01_VERSION:\n" \
    #                                         "\t\tfrom azure.keyvault.v2016_10_01 import models as implModels\n" \
    #                                         "\telse:\n" \
    #                                         "\t\traise NotImplementedError('APIVersion {} is not available'.format(api_version))\n" \
    #                                         "\treturn implModels"
    # elif (inputs["model"] == "Conv1d"):
    #     if inputs["utterance"].startswith("get_manifest should return an"):
    #         output["predicted_sql_query"] = "def player_move(board):\n" \
    #                                         "\t'''Shows the board to the player on the console and asks them to make a move.'''\n" \
    #                                         "\tprint(board, end='\n\n')\n" \
    #                                         "\tx, y = input('Enter move (e.g. 2b): ')\n" \
    #                                         "\tprint()\n" \
    #                                         "\treturn int(x) - 1, ord(y) - ord('a')\n"
    #     elif inputs["utterance"].startswith("Add routes by an resource"):
    #         output["predicted_sql_query"] = "def _get_unpatched(cls):\n" \
    #                                         "\twhile cls.__module__.startswith('setuptools'):\n" \
    #                                         "\t\tcls, = cls.__bases__\n" \
    #                                         "\tif not cls.__module__.startswith('distutils'):\n" \
    #                                         "\t\traise AssertionError(\n" \
    #                                         "\t\t\t'distutils has already been patched by %r' % cls\n" \
    #                                         "\t\t)\n" \
    #                                         "\treturn cls\n"
    # elif (inputs["model"] == "SelfAttn"):
    #     if inputs["utterance"].startswith("get_manifest should return an"):
    #         output["predicted_sql_query"] = "def get_manifest(self, repo_name, digest=None, version='v1'):\n" \
    #                                         "\taccepts = {'config': 'application/vnd.docker.container.image.v1+json',\n" \
    #                                         "\t\t\t'v1': 'application/vnd.docker.distribution.manifest.v1+json',\n" \
    #                                         "\t\t\t'v2': 'application/vnd.docker.distribution.manifest.v2+json'}\n" \
    #                                         "\turl = self._get_manifest_selfLink(repo_name, digest)\n" \
    #                                         "\tbot.verbose('Obtaining manifest: %s %s' % (url, version))\n" \
    #                                         "\theaders = {'Accept': accepts[version] }\n" \
    #                                         "\ttry:\n" \
    #                                         "\t\tmanifest = self._get(url, headers=headers, quiet=True)\n" \
    #                                         "\t\tmanifest['selfLink'] = url\n" \
    #                                         "\texcept:\n" \
    #                                         "\t\tmanifest = None\n" \
    #                                         "\treturn manifest\n"
    #     elif inputs["utterance"].startswith("Add routes by an resource"):
    #         output[
    #             "predicted_sql_query"] = "def add_resource_object(self, path: str, resource, methods: tuple=tuple(), names: Mapping=None):\n" \
    #                                      "\tnames = names or {}\n" \
    #                                      "\tif methods:\n" \
    #                                      "\t\tmethod_names = methods\n" \
    #                                      "\telse:\n" \
    #                                      "\t\tmethod_names = self.HTTP_METHOD_NAMES\n" \
    #                                      "\tfor method_name in method_names:\n" \
    #                                      "\t\thandler = getattr(resource, method_name, None)\n" \
    #                                      "\t\tif handler:\n" \
    #                                      "\t\t\tname = names.get(method_name, self.get_default_handler_name(resource, method_name))\n" \
    #                                      "\t\t\tself.add_route(method_name.upper(), path, handler, name=name)\n"

    if inputs['model'] == 'NBOW':
        model_name = 'nbow'
    elif inputs['model'] == 'BiRNN':
        model_name = 'birnn'
    elif inputs['model'] == 'Conv1d':
        model_name = 'conv1d_res'
    elif inputs['model'] == 'SelfAttn':
        model_name = 'self_attn'
    else:
        raise NotImplementedError
    model_path = f"~/ncc_data/codesearchnet/retrieval/data-mmap/all/{model_name}/checkpoints/checkpoint_best.pt"
    input = inputs['utterance']
    out = cli_main(model_path, input, kwargs='{"lang":["ruby"]}')
    output["predicted_sql_query"] = out[0][0]

    return json.dumps(output)


@app.route('/api/summarize', methods=['POST'])  # , methods=['POST']
# @cross_origin()
def summarize():
    inputs = request.get_data(as_text=True)
    # console.log(inputs)
    # input = inputs["sentence"]
    # tt = request.args.get('tt')
    # # re = {
    # # 'code': 0,
    # # 'data':'xxxxdata',
    # # 'message': "这是测试呀"
    # # }
    # model_input = ujson.loads(inputs)["code"]
    # model_path = '~/.ncc/demo/summarization/neural_transformer/python_wan.pt'
    # predicted_summary = cli_main(os.path.expanduser(model_path), input=model_input)
    #
    # # invoke our model API and obtain the output
    # output = {"predicted_summary": predicted_summary}
    # output["top_tokens"] = [['aaa', 'aaaaaaa'], ['bbb', 'bbbbbbb'], ['ccc', 'cccccc'], ['ddd', 'ddddd']]
    # output["top_indices"] = [[1, 2], [3,4], [5,6], [7,8]]
    # output["probabilities"] = [[0.11, 0.111], [0.22, 0.222], [0.33, 0.333], [0.44, 0.444]]

    output = {"predicted_summary": ''}
    inputs = json.loads(inputs)
    # if (inputs["model"] == "Transformer"):
    #     if inputs["code"].startswith("def _organize_states_for_post_update"):
    #         output[
    #             "predicted_summary"] = "make an initial pass across a set of states for update corresponding to post_update ."
    #     elif inputs["code"].startswith("def test_outdated_editables_columns_flag"):
    #         output["predicted_summary"] = "test the behavior of --editable --outdated flag in the list command .",
    #     elif inputs["code"].startswith("def translate_pattern"):
    #         output["predicted_summary"] = "translate a shell-like wildcard pattern to a compiled regular expression .",
    #     elif inputs["code"].startswith("def test_sobel_v_horizontal"):
    #         output["predicted_summary"] = "vertical sobel on a horizontal edge should be zero ."
    #     elif inputs["code"].startswith("def prewitt_h"):
    #         output["predicted_summary"] = "find the horizontal edges of an image using the prewitt transform ."
    # elif (inputs["model"] == "Seq2seq"):
    #     if inputs["code"].startswith("def _organize_states_for_post_update"):
    #         output["predicted_summary"] = "make an initial pass across a set of states for update within post_update ."
    #     elif inputs["code"].startswith("def test_outdated_editables_columns_flag"):
    #         output["predicted_summary"] = "test the behavior of --editable --uptodate flag in the list command .",
    #     elif inputs["code"].startswith("def translate_pattern"):
    #         output["predicted_summary"] = "translate a shell-like wildcard pattern to a regular expression pattern .",
    #     elif inputs["code"].startswith("def test_sobel_v_horizontal"):
    #         output["predicted_summary"] = "vertical scharr on a horizontal edge should be zero ."
    #     elif inputs["code"].startswith("def prewitt_h"):
    #         output["predicted_summary"] = "find the vertical edges of an image using the sobel transform ."
    # elif (inputs["model"] == "Tree2Seq"):
    #     if inputs["code"].startswith("def _organize_states_for_post_update"):
    #         output["predicted_summary"] = "make an initial pass across a set of states for update ."
    #     elif inputs["code"].startswith("def test_outdated_editables_columns_flag"):
    #         output["predicted_summary"] = "test the behavior of --editables flag in the list command .",
    #     elif inputs["code"].startswith("def translate_pattern"):
    #         output["predicted_summary"] = "translate a shell-like wildcard pattern to a regular expression; .",
    #     elif inputs["code"].startswith("def test_sobel_v_horizontal"):
    #         output["predicted_summary"] = "sobel on a horizontal edge should be a horizontal line ."
    #     elif inputs["code"].startswith("def prewitt_h"):
    #         output["predicted_summary"] = "find the horizontal edges of an image ."

    model_path = f"~/ncc_data/python_wan/summarization/data-mmap/{str.lower(inputs['model'])}/checkpoints/checkpoint_best.pt"
    input = inputs["code"]
    output["predicted_summary"] = cli_main(model_path, input)

    # rsp = flask.Response(json.dumps(output))
    # rsp.headers = headers
    # rsp.headers['Cache-Control'] = 'no-cache'
    # return rsp

    return json.dumps(output)

# if __name__=="__main__":
#     app.run(debug=False,host='0.0.0.0', port=5002)
