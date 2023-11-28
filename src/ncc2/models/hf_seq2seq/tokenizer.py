from transformers import AutoTokenizer, AutoConfig

from typing import Optional, final
from overrides import final as finaloverride
from ncc2.data.typing import PathLike

@final
class HFTokenizer(AutoTokenizer):
        
    def __init__(self, pathname: PathLike):
        self.model = AutoTokenizer.from_pretrained(pathname)
        
        self.vocab_info = AutoConfig.from_pretrained(pathname)
    
    @finaloverride    
    def create_encoder(self):
        return self.model.batch_encode_plus
    
    @finaloverride
    def create_decoder(self):
        return self.model.batch_decode