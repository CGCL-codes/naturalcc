from typing import Optional
from ncc2.nn.position_encoder import RotaryEncoder
from ncc2.typing import DataType, Device
from transformers import AutoModelForCausalLM,AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration
from ncc2.models.utils.arch_registry import ArchitectureRegistry
from dataclasses import dataclass
from ncc2.data import VocabularyInfo

@dataclass
class HFSeq2seqConfig:
    model_name: str

hf_seq2seq_archs = ArchitectureRegistry("hf_seq2seq")

hf_seq2seq_arch = hf_seq2seq_archs.marker

@hf_seq2seq_arch("auto")
def _auto() -> HFSeq2seqConfig:
    return HFSeq2seqConfig(
        model_name='auto'
    )

class HFSeq2seqBuilder:
    
    pos_encoder: Optional[RotaryEncoder]
    device: Optional[Device]
    dtype: Optional[DataType]
    
    def __init__(
        self,
        config: HFSeq2seqConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None
    ) -> None:
        self.model_name = config.model_name
        self.device, self.dtype = device,dtype
        
    def build_model(self):
        return AutoModelForSeq2SeqLM



