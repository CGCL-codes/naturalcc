from typing import Optional
from ncc2.nn.position_encoder import RotaryEncoder
from ncc2.typing import DataType, Device
from transformers import AutoModelForCausalLM
from transformers import T5ForConditionalGeneration
from ncc2.models.utils.arch_registry import ArchitectureRegistry
from dataclasses import dataclass
from ncc2.data import VocabularyInfo

@dataclass
class HFConfig:
    model_name: str

hf_archs = ArchitectureRegistry("hf_seq2seq")

hf_arch = hf_archs.marker

@hf_arch("auto")
def _auto() -> HFConfig:
    return HFConfig(
        model_name='auto'
    )

class HFBuilder:
    
    pos_encoder: Optional[RotaryEncoder]
    device: Optional[Device]
    dtype: Optional[DataType]
    
    def __init__(
        self,
        config: HFConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None
    ) -> None:
        self.model_name = config.model_name
        self.device, self.dtype = device,dtype
        
    def build_model(self):
        return T5ForConditionalGeneration



