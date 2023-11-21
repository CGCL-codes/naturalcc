from .generation import GenerationTask
from fairseq2.models.llama.builder import llama_archs
from fairseq2.tasks import register_task

@register_task('codellama_generation')
class CodellamaGenerationTask(GenerationTask):
    def load_state(self,ckpt_path):
        key_map = {}
        key_map['tok_embeddings.weight'] = 'decoder_frontend.embed.weight'
        key_map['norm.weight'] = 'decoder.layer_norm.weight'
        key_map['output.weight'] = 'final_proj.weight'
        for i in range(32):
            key_map[f'layers.{i}.attention_norm.weight'] = f'decoder.layers.{i}.self_attn_layer_norm.weight'
            key_map[f'layers.{i}.attention.wq.weight'] = f'decoder.layers.{i}.self_attn.q_proj.weight'
            key_map[f'layers.{i}.attention.wk.weight'] = f'decoder.layers.{i}.self_attn.k_proj.weight'
            key_map[f'layers.{i}.attention.wv.weight'] = f'decoder.layers.{i}.self_attn.v_proj.weight'
            key_map[f'layers.{i}.attention.wo.weight'] = f'decoder.layers.{i}.self_attn.output_proj.weight'
            key_map[f'layers.{i}.feed_forward.w1.weight'] = f'decoder.layers.{i}.ffn.gate_proj.weight'
            key_map[f'layers.{i}.feed_forward.w2.weight'] = f'decoder.layers.{i}.ffn.output_proj.weight'
            key_map[f'layers.{i}.feed_forward.w3.weight'] = f'decoder.layers.{i}.ffn.inner_proj.weight'
            key_map[f'layers.{i}.ffn_norm.weight'] = f'decoder.layers.{i}.ffn_layer_norm.weight'
        super().load_state(ckpt_path,key_map)
    
