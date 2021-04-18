# Test data

## Completion Task
1. SeqRNN: ```~/.ncc/demo/completion/seqrnn/py150.pt```

Source Codes & Ground-Truth
*completion only support code token inference*
```
"def loads(s, encoding=None, cls=None, object_hook=None, parse_float=None,\n        parse_int=None, parse_constant=None, **kw):\n    if (cls is None and encoding is None and object_hook is None and\n                parse_int is None and parse_float is None and\n                parse_constant is None and not kw):\n        return _default_decoder.decode(s)\n    if cls is None:\n        cls = JSONDecoder\n    if object_hook is not None:\n        kw['object_hook'] = object_hook\n    if parse_float is not None:\n        kw['parse_float'] = parse_float\n    if parse_int is not None:\n        kw['parse_int'] = parse_int\n    if parse_constant is not None:\n        kw['parse_constant'] = parse_constant\n    return"
>> "cls(encoding=encoding, **kw).decode(s)"

"body_content = self._serialize.body(parameters, 'ServicePrincipalCreateParameters')\nrequest = self._client.post(url, query_parameters)\nresponse = self._client.send( request, header_parameters, body_content, operation_config)"
>> "(request, header_parameters, body_content, **operation_config)"

"create_train_gen = lambda: data_loader.create_random_gen()\ncreate_eval_train_gen = lambda: data_loader.create_fixed_gen(\"train\")\ncreate_eval_valid_gen = lambda: data_loader.create_fixed_gen(\"valid\")\ncreate_eval_test_gen = lambda: data_loader.create_fixed_gen"
>> "data_loader.create_fixed_gen("test")"

"def build_model():\n    l0 = nn.layers.InputLayer((batch_size, data.num_classes))\n\n    l0_size = nn.layers.InputLayer((batch_size, 52))\n    l1_size = nn.layers.DenseLayer(l0_size, num_units=80, W=nn_plankton.Orthogonal('relu'), b=nn.init.Constant(0.1))\n    l2_size = nn.layers.DenseLayer(l1_size, num_units=80, W=nn_plankton.Orthogonal('relu'), b=nn.init.Constant(0.1))\n    l3_size = nn.layers.DenseLayer(l2_size, num_units=data.num_classes, W=nn_plankton.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=None)\n\n    l1 = nn_plankton.NonlinLayer(l0, T.log)\n    ltot = nn.layers.ElemwiseSumLayer([l1, l3_size])\n\n    lout ="
>> "nn_plankton.NonlinLayer(ltot, nonlinearity=T.nnet.softmax)"

"if self.query is not None:\n  oprot.writeFieldBegin('query', TType.STRING, 1)\n  oprot.writeString(self.query)\n  oprot.writeFieldEnd()\nif self.configuration is not None:\n  oprot.writeFieldBegin('configuration', TType.LIST, 3)\n  oprot.writeListBegin(TType.STRING, len(self.configuration))\n  for iter6 in self.configuration:\n    oprot.writeString(iter6)\n  oprot.writeListEnd()\n  oprot.writeFieldEnd()\nif self.hadoop_user is not None:\n  oprot.writeFieldBegin('hadoop_user', TType.STRING, 4)\n  oprot"
>> "writeString(self.hadoop_user)\noprot.writeFieldEnd()"

"location = module.db_location\nif location is not None:\n    childNode = ElementTree.SubElement(node, 'location')\n    self.getDao('location').toXML(location, childNode)\nfunctions = module.db_functions\nfor function in functions:\n    childNode = ElementTree.SubElement(node, 'function')\n    self.getDao('function').toXML(function, childNode)\nannotations = module.db_annotations\nfor annotation in annotations:\n    childNode = ElementTree.SubElement(node, 'annotation')\n    self.getDao('annotation').toXML(annotation, childNode)\nportSpecs = module.db_portSpecs\nfor portSpec in portSpecs:"
>> "childNode = ElementTree.SubElement(node, 'portSpec')\nself.getDao('portSpec').toXML(portSpec, childNode)"
```



## Summarization Task
1. Neural-Transformer: ```~/.ncc/demo/summarization/neural_transformer/python_wan.pt```
2. Seq2Seq: ```~/.ncc/demo/summarization/seq2seq/python_wan.pt```

Source Codes & Ground-Truth
```
"def positional(max_positional_args):\n\tdef positional_decorator(wrapped):\n\t\t@functools.wraps(wrapped)\n\t\tdef positional_wrapper(*args, **kwargs):\n\t\t\tif (len(args) > max_posi      tional_args):\n\t\t\t\tplural_s = ''\n\t\t\t\tif (max_positional_args != 1):\n\t\t\t\t\tplural_s = 's'\n\t\t\t\tmessage = ('%s()\ttakes\tat\tmost\t%d\tpositional\targument%s\t(%d\tgive      n)' % (wrapped.__name__, max_positional_args, plural_s, len(args)))\n\t\t\t\tif (positional_parameters_enforcement == POSITIONAL_EXCEPTION):\n\t\t\t\t\traise TypeError(message)\n\t\t\t      \telif (positional_parameters_enforcement == POSITIONAL_WARNING):\n\t\t\t\t\tlogger.warning(message)\n\t\t\t\telse:\n\t\t\t\t\tpass\n\t\t\treturn wrapped(*args, **kwargs)\n\t\treturn p      ositional_wrapper\n\tif isinstance(max_positional_args, six.integer_types):\n\t\treturn positional_decorator\n\telse:\n\t\t(args, _, _, defaults) = inspect.getargspec(max_positional_ar      gs)\n\t\treturn positional((len(args) - len(defaults)))(max_positional_args)"
>> "a decorator to declare that only the first n arguments my be positional ."

"def getCarveIntersectionFromEdge(edge, vertexes, z):\n\tfirstVertex = vertexes[edge.vertexIndexes[0]]\n\tfirstVertexComplex = firstVertex.dropAxis(2)\n\tsecondVertex = vertexes[edge.v      ertexIndexes[1]]\n\tsecondVertexComplex = secondVertex.dropAxis(2)\n\tzMinusFirst = (z - firstVertex.z)\n\tup = (secondVertex.z - firstVertex.z)\n\treturn (((zMinusFirst * (secondVerte      xComplex - firstVertexComplex)) \/ up) + firstVertexComplex)"
>> "get the complex where the carve intersects the edge ."

"def MessageEncoder(field_number, is_repeated, is_packed):\n\ttag = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)\n\tlocal_EncodeVarint = _EncodeVarint\n\tassert (not i      s_packed)\n\tif is_repeated:\n\t\tdef EncodeRepeatedField(write, value):\n\t\t\tfor element in value:\n\t\t\t\twrite(tag)\n\t\t\t\tlocal_EncodeVarint(write, element.ByteSize())\n\t\t\t      \telement._InternalSerialize(write)\n\t\treturn EncodeRepeatedField\n\telse:\n\t\tdef EncodeField(write, value):\n\t\t\twrite(tag)\n\t\t\tlocal_EncodeVarint(write, value.ByteSize())\n\      t\t\treturn value._InternalSerialize(write)\n\t\treturn EncodeField"
>> "returns an encoder for a message field ."

"def getRadialPath(begin, center, end, path):\n\tbeginComplex = begin.dropAxis()\n\tendComplex = end.dropAxis()\n\tcenterComplex = center.dropAxis()\n\tbeginMinusCenterComplex = (beginComplex - centerComplex)\n\tendMinusCenterComplex = (endComplex - centerComplex)\n\tbeginMinusCenterComplexRadius = abs(beginMinusCenterComplex)\n\tendMinusCenterComplexRadius = abs(endMinusCenterComplex)\n\tif ((beginMinusCenterComplexRadius == 0.0) or (endMinusCenterComplexRadius == 0.0)):\n\t\treturn [begin]\n\tbeginMinusCenterComplex \/= beginMinusCenterComplexRadius\n\tendMinusCenterComplex \/= endMinusCenterComplexRadius\n\tangleDifference = euclidean.getAngleDifferenceByComplex(endMinusCenterComplex, beginMinusCenterComplex)\n\tradialPath = []\n\tfor point in path:\n\t\tweightEnd = point.x\n\t\tweightBegin = (1.0 - weightEnd)\n\t\tweightedRadius = ((beginMinusCenterComplexRadius * weightBegin) + ((endMinusCenterComplexRadius * weightEnd) * (1.0 + point.y)))\n\t\tradialComplex = ((weightedRadius * euclidean.getWiddershinsUnitPolar((angleDifference * point.x))) * beginMinusCenterComplex)\n\t\tpolygonPoint = (center + Vector3(radialComplex.real, radialComplex.imag, point.z))\n\t\tradialPath.append(polygonPoint)\n\treturn radialPath"
>> "get radial path ."

"def compare_package(version1, version2):\n\tdef normalize(v):\n\t\treturn [int(x) for x in re.sub('(\\\\.0+)*$', '', v).split('.')]\n\treturn cmp(normalize(version1), normalize(version2))"
>> "compare version packages ."

"def _get_hub():\n\tglobal _threadlocal\n\ttry:\n\t\treturn _threadlocal.hub\n\texcept AttributeError:\n\t\tpass"
>> "return the hub for the current thread ."

"@environmentfilter\ndef do_attr(environment, obj, name):\n\ttry:\n\t\tname = str(name)\n\texcept UnicodeError:\n\t\tpass\n\telse:\n\t\ttry:\n\t\t\tvalue = getattr(obj, name)\n\t\texcept AttributeError:\n\t\t\tpass\n\t\telse:\n\t\t\tif (environment.sandboxed and (not environment.is_safe_attribute(obj, name, value))):\n\t\t\t\treturn environment.unsafe_undefined(obj, name)\n\t\t\treturn value\n\treturn environment.undefined(obj=obj, name=name)"
>> "get an attribute of an object ."

"def mail_managers(subject, message, fail_silently=False, connection=None):\n\tif (not settings.MANAGERS):\n\t\treturn\n\tEmailMessage((u'%s%s' % (settings.EMAIL_SUBJECT_PREFIX, subject)), message, settings.SERVER_EMAIL, [a[1] for a in settings.MANAGERS], connection=connection).send(fail_silently=fail_silently)"
>> "sends a message to the managers ."

"def aliased(element, alias=None, name=None, flat=False, adapt_on_names=False):\n\tif isinstance(element, expression.FromClause):\n\t\tif adapt_on_names:\n\t\t\traise sa_exc.ArgumentError('adapt_on_names\tonly\tapplies\tto\tORM\telements')\n\t\treturn element.alias(name, flat=flat)\n\telse:\n\t\treturn AliasedClass(element, alias=alias, flat=flat, name=name, adapt_on_names=adapt_on_names)"
>> "produce an alias of the given element ."

"def preserve_value(namespace, name):\n\tdef decorator(func):\n\t\tdef resetter_attr(saved_value_internal):\n\t\t\treturn setattr(namespace, name, saved_value_internal)\n\t\tdef resetter_no_attr(saved_value_internal):\n\t\t\tdel saved_value_internal\n\t\t\treturn delattr(namespace, name)\n\t\tdef wrapper(*args, **kwargs):\n\t\t\tsaved_value = None\n\t\t\ttry:\n\t\t\t\tsaved_value = getattr(namespace, name)\n\t\t\t\tresetter = resetter_attr\n\t\t\texcept AttributeError:\n\t\t\t\tresetter = resetter_no_attr\n\t\t\ttry:\n\t\t\t\treturn func(*args, **kwargs)\n\t\t\tfinally:\n\t\t\t\tresetter(saved_value)\n\t\twrapper.__name__ = func.__name__\n\t\twrapper.__doc__ = func.__doc__\n\t\treturn wrapper\n\treturn decorator"
>> "function decorator to wrap a function that sets a namespace item ."
```


## Completion Task
Coming Soon




