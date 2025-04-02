
import os
import io
import hashlib
import torch
from datasets import Dataset, load_dataset
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format
from tqdm import tqdm
from utils import *
import time

model_name = 'VLM_WebSight_finetuned'
test_data_name = 'websight'

model_path = f'/data02/projects/vision2ui/models/{model_name}'
result_path = f'/data02/projects/vision2ui/results/{model_name}-{test_data_name}'
data_path = f'/data02/projects/vision2ui/datasets/{test_data_name}_benchmark'

DEVICE = torch.device("cuda:5")
PROCESSOR = AutoProcessor.from_pretrained(
    model_path,
)
MODEL = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(DEVICE)
image_seq_len = MODEL.config.perceiver_config.resampler_n_latents
BOS_TOKEN = PROCESSOR.tokenizer.bos_token
BAD_WORDS_IDS = PROCESSOR.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids


def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

def custom_transform(x):
    x = convert_to_rgb(x)
    x = to_numpy_array(x)
    x = resize(x, (960, 960), resample=PILImageResampling.BILINEAR)
    x = PROCESSOR.image_processor.rescale(x, scale=1 / 255)
    x = PROCESSOR.image_processor.normalize(
        x,
        mean=PROCESSOR.image_processor.image_mean,
        std=PROCESSOR.image_processor.image_std
    )
    x = to_channel_dimension_format(x, ChannelDimension.FIRST)
    x = torch.tensor(x)
    return x

def predict(image):
    inputs = PROCESSOR.tokenizer(
        f"{BOS_TOKEN}<fake_token_around_image>{'<image>' * image_seq_len}<fake_token_around_image>",
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs["pixel_values"] = PROCESSOR.image_processor([image], transform=custom_transform)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    generated_ids = MODEL.generate(**inputs, bad_words_ids=BAD_WORDS_IDS, max_length=4096)
    generated_text = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

if test_data_name == 'vision2ui':
    ds = load_dataset('parquet',data_dir=data_path)['train']
else:
    ds = Dataset.load_from_disk(data_path)

for item in tqdm(ds):
    image = item['image']
    md5 = image2md5(image)
    save_path = result_path
    if test_data_name == 'vision2ui':
        tokens = sum(item['tokens'])
        size = 'short' if tokens<2048 else ('mid' if tokens<4096 else 'long')
        save_path = os.path.join(result_path,size)

    if os.path.exists(os.path.join(save_path,md5)):
        continue

    t_start=time.time()
    html = predict(image)

    duration = time.time()-t_start
    save_result(save_path, image, item['text'] if 'text' in item else item['html'], html, duration)


