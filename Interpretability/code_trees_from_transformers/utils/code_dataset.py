from code_yk import get_actions,get_nonbinary_spans,get_tags_tokens
import json
symbol=['(', ')', ':', '{', '.', ',', '}', '=', '[', ']', '*', '+', '-', '>=', '**', '>', '==', '/', '<', '<=', '!=', '%', '+=',
        '>>', '*=', '-=', '_', '//', '->', '<<', '|=', '&','~', '/=', '__', '@', '|', '...', '^=', '>>=', 'id_', '&=', '^', '//=', '<<=', '%=', '**=']
class Dataset(object):
    def __init__(self, path, tokenizer):
        self.path = path
        self.tokenizer = tokenizer
        self.cnt = 0
        self.sents = []
        self.raw_tokens = []
        self.tokens = []
        # self.masks = []
        self.gold_spans = []
        self.gold_tags = []
        self.gold_trees = []
        self.raw_tokens_text=[]
        self.raw_tokens_text_number=[]

        # flatten = lambda l: [item for sublist in l for item in sublist]

        with open(path, 'r') as f:
            lines = f.readlines()

        dict_ast = []  # 包含ast的列表
        for dict in lines:
            self_dict = json.loads(dict)
            dict_ast.append(self_dict)

        for single_ast in dict_ast:
            ast=single_ast['ast']
            raw_tokens=single_ast['code_tokens']
            raw_tokens_text=[i for i in raw_tokens if i not in symbol] #把符号去掉
            raw_tokens_text_number=[i for i in range(len(raw_tokens)) if raw_tokens[i] not in symbol]
            sent = ' '.join(raw_tokens)
            actions = get_actions(ast)
            self.cnt += 1
            self.sents.append(sent)
            self.raw_tokens.append(raw_tokens)
            self.tokens.append(self.tokenizer.tokenize(sent))
            gold_spans, gold_tags, _, _ = get_nonbinary_spans(actions)
            self.gold_spans.append(gold_spans)
            self.gold_tags.append(gold_tags)
            self.gold_trees.append(ast)
            self.raw_tokens_text.append(raw_tokens_text)
            self.raw_tokens_text_number.append(raw_tokens_text_number)