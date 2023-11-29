import torch

from ncc import LOGGER
from ncc import tasks
from ncc.utils import utils
from ncc.utils.checkpoint_utils import load_checkpoint_to_cpu
from ncc.utils.file_ops.yaml_io import recursive_expanduser, recursive_contractuser


def main(model_path, input):
    LOGGER.info('Load model from {}'.format(model_path))
    state = load_checkpoint_to_cpu(model_path, arg_overrides={})
    args = state["args"]
    args = recursive_contractuser(args, old_cache_name='.ncc')
    args = recursive_expanduser(args)
    task = tasks.setup_task(args)  # load src/tgt dicts
    model = task.build_model(args)
    model.load_state_dict(state["model"])
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.set_device(torch.cuda.device_count() - 1)
        model.cuda()
    model.eval()
    if args['common']['fp16'] and use_cuda:
        model.half()

    sample = task.encode_input(input)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    generator = task.sequence_completor
    net_output = generator.complete(models=[model], sample=sample)
    out = task.decode_output(net_output)
    return out


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Command Interface")
    parser.add_argument(
        "--model", "-m", type=str, help="pytorch model path",
    )
    parser.add_argument(
        "--input", "-i", type=str, help="model input",
    )
    args = parser.parse_args()

    code_tokens = """
    def loads(s, encoding=None, cls=None, object_hook=None, parse_float=None,
            parse_int=None, parse_constant=None, **kw):
        if (cls is None and encoding is None and object_hook is None and
                    parse_int is None and parse_float is None and
                    parse_constant is None and not kw):
            return _default_decoder.decode(s)
        if cls is None:
            cls = JSONDecoder
        if object_hook is not None:
            kw['object_hook'] = object_hook
        if parse_float is not None:
            kw['parse_float'] = parse_float
        if parse_int is not None:
            kw['parse_int'] = parse_int
        if parse_constant is not None:
            kw['parse_constant'] = parse_constant
        return
    """
    # cls(encoding=encoding, **kw).decode(s)
    code_tokens = """
    body_content = self._serialize.body(parameters, 'ServicePrincipalCreateParameters')
    request = self._client.post(url, query_parameters)
    response = self._client.send
        """.strip()
    # request, header_parameters, body_content, operation_config
    code_tokens = """
    create_train_gen = lambda: data_loader.create_random_gen()
    create_eval_train_gen = lambda: data_loader.create_fixed_gen("train")
    create_eval_valid_gen = lambda: data_loader.create_fixed_gen("valid")
    create_eval_test_gen = lambda:
        """.strip()
    # data_loader.create_fixed_gen("test")
    code_tokens = """
    def build_model():
        l0 = nn.layers.InputLayer((batch_size, data.num_classes))

        l0_size = nn.layers.InputLayer((batch_size, 52))
        l1_size = nn.layers.DenseLayer(l0_size, num_units=80, W=nn_plankton.Orthogonal('relu'), b=nn.init.Constant(0.1))
        l2_size = nn.layers.DenseLayer(l1_size, num_units=80, W=nn_plankton.Orthogonal('relu'), b=nn.init.Constant(0.1))
        l3_size = nn.layers.DenseLayer(l2_size, num_units=data.num_classes, W=nn_plankton.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=None)

        l1 = nn_plankton.NonlinLayer(l0, T.log)
        ltot = nn.layers.ElemwiseSumLayer([l1, l3_size])

        lout =  
        """.strip()
    # nn_plankton.NonlinLayer(ltot, nonlinearity=T.nnet.softmax)
    code_tokens = """
    if self.query is not None:
      oprot.writeFieldBegin('query', TType.STRING, 1)
      oprot.writeString(self.query)
      oprot.writeFieldEnd()
    if self.configuration is not None:
      oprot.writeFieldBegin('configuration', TType.LIST, 3)
      oprot.writeListBegin(TType.STRING, len(self.configuration))
      for iter6 in self.configuration:
        oprot.writeString(iter6)
      oprot.writeListEnd()
      oprot.writeFieldEnd()
    if self.hadoop_user is not None:
      oprot.writeFieldBegin('hadoop_user', TType.STRING, 4)
      oprot
        """.strip()
    # writeString(self.hadoop_user)
    # oprot.writeFieldEnd()

    code_tokens = """
    location = module.db_location
    if location is not None:
        childNode = ElementTree.SubElement(node, 'location')
        self.getDao('location').toXML(location, childNode)
    functions = module.db_functions
    for function in functions:
        childNode = ElementTree.SubElement(node, 'function')
        self.getDao('function').toXML(function, childNode)
    annotations = module.db_annotations
    for annotation in annotations:
        childNode = ElementTree.SubElement(node, 'annotation')
        self.getDao('annotation').toXML(annotation, childNode)
    portSpecs = module.db_portSpecs
    for portSpec in portSpecs:
        """.strip()

    # import re
    # code_tokens = code_tokens.replace('lambda', ' ').replace('if', ' ').replace('is', ' ').replace('not', ' ')
    # code_tokens = re.split(r'[\s|\.|)|(|,|:]+', code_tokens)
    # code_tokens = [token for token in code_tokens if len(token) > 1]
    # print(code_tokens)
    # # exit()

    args.input = code_tokens
    out = main(args.model, args.input)
    print(out)


if __name__ == '__main__':
    cli_main()
