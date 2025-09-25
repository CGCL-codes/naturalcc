from datasets import load_dataset,Dataset
from tqdm import tqdm
import json
import sys, os
from PIL import Image
from pathlib import Path
import argparse
from datetime import datetime
import traceback
from typing import Iterable
from bs4 import BeautifulSoup as bs
import random
import numpy as np

sys.path.append(os.path.abspath('.'))
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import agents
from html2shot_sync import html2shot
from utils.utils import *
from smart_blocker import blocker
from vendors.google__ import gemini
from vendors.openai__ import gpt4o
from vendors.deepseek__ import deepseek_vl2, init_llm
from agents import *
from utils.log import init_logger, logger
from evaluation.mrweb.study import mae_score
from evaluation.metrics import clip_sim
from my_datasets import*

DEBUG = False
SEED = 2026
MAX_BLOCKS_LIMIT = 25 # 

def set_global_seed(seed: int):
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    agents.SEED = SEED
    
set_global_seed(SEED)
    
BACKBONES = [gpt4o, gemini, deepseek_vl2]

AGENT_Generate = AgentGenerate()
AGENT_GenerateElf = AgentGenerateElf()
AGENT_Assemble = AgentAssemble() 
AGENT_GetText = AgentGetText() 
AGENT_Refine = AgentRefine()


def refine(design:Image.Image, pred_img:Image.Image, pred_code:str):
    text_content = AGENT_GetText.infer(imgs=[design])
    user_text =json.dumps(
    {
        "task": "Based on the design image (Image 1) and the screenshot of the generated webpage (Image 2), \
        combined with the textual_content extracted from the original webpage, \
        refine the webpage_code according to the requirements in the prompt.",
        "webpage_code": pred_code,
        "textual_content":text_content,
    }
    )
    
    refined_code = AGENT_Refine.infer(imgs=[design,pred_img],text=user_text)
    return refined_code

# 
def normalize_value(value, min_value, max_value, reverse=False):
    normalized_value = (value - min_value) / (max_value - min_value)
    if reverse:
        # （）， 1 - normalized_value
        return 1 - normalized_value
    return normalized_value

def evaluate_images(img1, img2):
    """
    ，。
    """

    #clip_scorer = CLIPScorer(model_name='ViT-B-32-quickgelu', pretrained='openai')
    #lpips_scorer = LPIPSScorer(net='vgg')

    #clip_score = clip_scorer.score(img1, img2)
    #lpips_score = lpips_scorer.score(img1, img2)
    #ssim_value = ssim_score(img1, img2)
    #psnr_value = psnr_score(img1, img2)
    mae_value = mae_score(img1, img2)
    # emd_score = emd_similarity(img1, img2, max_size=96, mode="RGB")
    clip_value = clip_sim(img1, img2)

    return mae_value, clip_value
  

def verify_score(mae, clip_similarity,  weights=(0.5, 0.5)):
    """
    Computes a composite similarity score based on MAE, NEMD, and CLIP similarity.

    Args:
        mae (float): Mean Absolute Error between images, normalized.
        nemd (float): 1-emd, and Normalized Earth Mover's Distance (already in [0, 1]).
        clip_similarity (float): CLIP cosine similarity between images.
        weights (tuple): Weights for (MAE_similarity, NEMD_similarity, CLIP_transformed).

    Returns:
        float: Composite similarity score.
    """
    w1, w2 = weights    
    
    # Compute composite score
    # clip，
    composite_score = w1 * (1 - mae/255)  + w2 * (clip_similarity ** 0.5)
    
    return composite_score
     

def get_best(ref_img, cand_imgs):
    assert len(cand_imgs)   >1,     "There must be more than one candidates to select."  
    
    scores = {'MAE':[], 'CLIP':[]}
    for cand_image in cand_imgs:
        mae_value, clip_value = evaluate_images(ref_img, cand_image)
        scores['MAE'].append(mae_value)
        scores['CLIP'].append(clip_value)
            
    # MAEEMD，
    #all_mae = [entry['scores']['MAE'] for entry in scores['MAE']]
    
    #min_mae = min(all_mae)
    #max_mae = max(all_mae)
    weights = (0.5, 0.5)
    final_scores = [verify_score(scores['MAE'][i], scores['CLIP'][i], weights) for i in range(len(cand_imgs))]
    logger.info(f"scores: {final_scores}")
    max_score = max(final_scores)
    idx = final_scores.index(max_score)
    
    return idx, final_scores  


def generate_module_code(image:Image.Image, plans, save_dir:Path, samples, temperature):
    codes_plans = []
    
    if DEBUG:
        tmp_dir = save_dir/ 'tmp'
        tmp_dir.mkdir(exist_ok=True, parents=True)
    
    for index,plan in tqdm(enumerate(plans,start=1),total=len(plans)):
        module_image = crop_image(image,plan)
        generator = AGENT_GenerateElf if agents.BACKBONE==deepseek_vl2 else AGENT_Generate # deepseekprompt

        try: #
            codes = generator.infer([module_image], n=samples, temperature= 0.0 if samples ==1 else temperature)
            if samples > 1: # do sampling
                imgs = []
                for j, c in enumerate(codes):
                    c = remove_code_markers(c)
                    pred_img = html2shot(c)
                    imgs.append(pred_img)
                    if DEBUG:
                        img_path = tmp_dir / f'{index}_{j}.png'
                        pred_img.save(str(img_path))
                        
                best_id, scores = get_best(module_image, imgs)
                logger.info(f"best id of module {index}: {best_id}, scores: {scores}")
                code = codes[best_id]
            else:            
                code=remove_code_markers(codes)                
            codes_plans.append({"module_position":plan,"module_code":code})
            
            if DEBUG:
                img_path = tmp_dir / f'{index}.png'
                module_image.save(str(img_path))
                pred_img = html2shot(code)
                pred_img.save(str(tmp_dir/f'{index}_pred.png'))
                with open(tmp_dir/f'{index}.html', 'w') as f:
                    f.write(code)
        except:
            logger.error(f"Fail to generate block {index}")
            continue
    
    return codes_plans

# 3
def agent_assemble(target_img, codes_plans, samples, temperature):
    html_image = []
    for _ in range(samples):
        html = AGENT_Assemble.infer([json.dumps(codes_plans), target_img], temperature= 0.0 if samples==1 else temperature)
        imageofhtml = html2shot(html_content=html)
        html_image.append({"html":html,"image":imageofhtml})

    return html_image

def absolute_assemble(image,code_plans):
    html_image = []
    #  HTML 
    framework_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    </head>
    <body>
    </body>
    </html>
    """

    #  HTML 
    framework_soup = bs(framework_html, 'html.parser')
    new_body = framework_soup.find('body')

    for node in code_plans:
        #  HTML  <body>  <div>,body，head
        soup = bs(node['module_code'], 'html.parser')
        body_tag = soup.find('body')
        if body_tag:
            body_tag.name = "div"  # 
        
        bbox = node['module_position']
        code_abs = f'<div style="position: absolute; overflow: hidden; border: 1px solid white; left: {round(bbox[0] * image.width)}px; top: {round(bbox[1] * image.height)}px; width: {round((bbox[2] - bbox[0]) * image.width)}px; height: {round((bbox[3] - bbox[1]) * image.height)}px;">{"".join(str(body_tag))}</div>'
        new_content = bs(code_abs, 'html.parser')
        new_body.append(new_content)
        #code_abss.append(code_abs)

    html = framework_soup.prettify()
    imageofhtml = html2shot(html_content=html)
    html_image.append({"html":html,"image":imageofhtml})
    
    return html_image


def save_result(save_dir:Path, target_image:Image.Image, target_html, pred_image:Image.Image, pred_html):
    target_image.save(str(save_dir/'answer.png'))
    pred_image.save(str(save_dir/'prediction.png'))
    
    with open(save_dir/'answer.html', 'w') as f:
        f.write(target_html)

    with open(save_dir/'prediction.html', 'w') as f:
        f.write(pred_html)

def pipeline(design_image, target_html, save_dir, generate_samples, agent_assembly_samples, sample_temperature):    
    logger.info("+++++++++++++++++Spliting the design image info pieces ...")
    # 
    plans ,blocks_image = blocker(design_image)

    # 
    blocks_image.save(str(save_dir/'blocks.png'))
    
    if len(plans) > MAX_BLOCKS_LIMIT:
        logger.error(f"Exceed the MAX_BLOCKS_LIMIT, {len(plans)} blocks.")
        raise ValueError("Too many blocks!")

    # 
    logger.info("+++++++++++++++++Generating the code of every piece ...")
    codes = generate_module_code(design_image, plans, save_dir, generate_samples, sample_temperature)
    

    
    pred_img = None
    pred_html = None

    # 1. 
    logger.info("+++++++++++++++++Assembling the code using absolute positioning ...")
    assemble_res_abs = absolute_assemble(design_image,codes)
        
    assemble_res_agent = []
    if agents.BACKBONE != deepseek_vl2: # ，agent
        # 2. agent
        logger.info("+++++++++++++++++Assembling the code with agent ...")
        assemble_res_agent = agent_assemble(design_image,codes, agent_assembly_samples, sample_temperature)
    
        # verifier
        logger.info("+++++++++++++++++Choosing the the best one ...")
        all_best_idx, scores = get_best(design_image, [r['image'] for r in assemble_res_agent + assemble_res_abs])
        agent_best_idx = scores.index(max(scores[:-1]))
        logger.info(f'The scores: scores, all best id: {all_best_idx}, agent best id: {agent_best_idx}.')
        pred_img = assemble_res_agent[agent_best_idx]['image']
        pred_html = assemble_res_agent[agent_best_idx]['html']
    
        if DEBUG:
            for i, item in enumerate(assemble_res_agent):
                img_path = save_dir / f'assemble_agent_{i}_{scores[i]}.png'
                item['image'].save(str(img_path))
                with open(save_dir/f'assemble_agent_{i}.html', 'w') as f:
                    f.write(item['html'])
            img_path = save_dir / f'assemble_abs_{scores[-1]}.png'
            assemble_res_abs[0]['image'].save(str(img_path))
            with open(save_dir/f'assemble_abs.html', 'w') as f:
                f.write(assemble_res_abs[0]['html'])
    else:
        pred_img = assemble_res_abs[0]['image']
        pred_html = assemble_res_abs[0]['html']
        
    """ 
    if DEBUG:
        img_path = save_dir / f'assemble_best.png'
        assemble_res[best_idx]['image'].save(str(img_path))
        with open(save_dir/'assemble_best.html', 'w') as f:
            f.write(assemble_res[best_idx]['html'])

    # refine
    logger.info("+++++++++++++++++Refining the code with agents ...")
    html_refined = refine(design_image, assemble_res[best_idx]['image'], assemble_res[best_idx]['html'])
    img_refined = html2shot(html_refined)
    """
 
    save_result(save_dir, design_image, target_html, pred_img, pred_html)

def main(ds:BaseDataset, out_dir:Path, generate_samples, agent_assembly_samples, sample_seed):
    for i, item in  enumerate(ds):
        md5 = image2md5(item['image'])
        save_dir = out_dir / md5
        save_dir.mkdir(exist_ok=True, parents= True)
        logger.info(f'+++++++++++++++++Generate the {i}th sample, md5:{md5}.+++++++++++++++++')
        try:
            pipeline(item['image'], item['text'], save_dir, generate_samples, agent_assembly_samples, sample_seed)
        except Exception as e:
            exception_info = traceback.format_exc()
            logger.error(f"EEORR:{e}\nTrace:\n{exception_info}")
            continue
  
DATASETS = [D2CHardDataset, V2UDataset, V2UDataset_old,tmpDataset]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=int, default=0, choices= list(range(len(DATASETS)))) # see DATASETS, 0: d2c, 1: v2u, 2:V2UDataset_old 
    parser.add_argument("--out_dir", "-o", type=str, default="./output")
    parser.add_argument("--backbone", "-b", type=int, default=0, choices=list(range(len(BACKBONES)))) #see BACKBONES, 0: gpt-4o, 1: Gemini ,2: deepseek_vl2
    parser.add_argument("--range", "-r", type=str, default='') # ，，'10_100'=[10, 100), '5'=[5, -1]
    parser.add_argument("--generate_samples", "-gs", type=int, default=1)
    parser.add_argument("--assembly_samples", "-as", type=int, default=1)
    parser.add_argument("--sample_temperature", "-st", type=float, default=0.0)
    parser.add_argument("--debug", "-db", type=bool, default=True)
    args=parser.parse_args()    

    
    # samplingdeepseek-vl2
    assert not (args.backbone !=2 and args.generate_samples > 1), "Generation sampling only is supported by deepseek-vl2."
    
    if args.backbone ==2: # deepseek-vl2 via vllm
        init_llm(torch.cuda.device_count(), SEED, 1)
    
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    save_dir=Path(args.out_dir) / time_str
    save_dir.mkdir(exist_ok=True, parents=True)
    
    init_logger(logger, logfile=str(save_dir/'log.txt'))  
    logger.info(args)
    
    # base MLLM
    agents.BACKBONE=BACKBONES[args.backbone]  # (prompt, text, [img1, ...])   
    DEBUG = args.debug
    
    range_ids = None
    if len(args.range.strip()) >0:
        res = args.range.strip().split('_')
        range_ids = [int(res[0]), int(res[1]) if len(res) == 2 else -1 ]
    ds = DATASETS[args.dataset](range_ids)    

    main(ds, save_dir, args.generate_samples, args.assembly_samples, args.sample_temperature)
    
    logger.info("All tasks have been done.")

