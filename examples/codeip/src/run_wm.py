from watermarking.wm import main, WmBaseArgs
from transformers import HfArgumentParser

parser = HfArgumentParser((WmBaseArgs,))
args: WmBaseArgs
args, = parser.parse_args_into_dataclasses()
main(args)