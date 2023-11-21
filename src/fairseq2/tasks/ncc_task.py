from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.utils.checkpoint_loader import load_checkpoint,upgrade_fairseq_checkpoint,convert_model_state_dict
from fairseq2.nn.position_encoder import RotaryEncoder
import torch
import os

class NccTask():
    @classmethod
    def __init__(self,archs:ArchitectureRegistry,model_name:str,builder,device=None):
        if not model_name in archs.names():
            raise ValueError("Arch {} don't includes model {}".format(archs.model_type,model_name))
        if not device:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("CUDA is available. Using GPU.")
            else:
                device = torch.device("cpu")
                print("CUDA is not available. Using CPU.")
        self.device = device
        self.load_model(archs,model_name,builder,device)
        
    @classmethod  
    def load_model(self,archs:ArchitectureRegistry,model_name:str,builder,device):
        self.model_name = model_name
        self.config = archs.get_config(model_name)
        self.builder = builder(self.config,device=device)
        self.model = self.builder.build_model()
        
    @classmethod    
    def load_state(self,ckpt_folder,key_map={}):
        ckpt_files = [f for f in os.listdir(ckpt_folder) if f.endswith('.pth')]
        ckpt = {}
        if not ckpt_files:
            raise FileExistsError('No *.pth found in {}'.format(ckpt_folder))
        for ckpt_file in ckpt_files:
            ckpt_path = os.path.join(ckpt_folder, ckpt_file)
            checkpoint = torch.load(ckpt_path)
            for key in checkpoint:
                ckpt[key] = checkpoint[key]
        ckpt2 = {}
        for key in key_map:
            ckpt2[key_map[key]] = ckpt[key]
        del ckpt
        self.model.load_state_dict(ckpt2)

    @classmethod
    def set_tokenizer(self,tokenizer):
        self.tokenizer = tokenizer

        
