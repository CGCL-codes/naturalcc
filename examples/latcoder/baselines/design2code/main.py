from datetime import datetime
import random
import os,sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import traceback
import argparse
sys.path.append(os.path.abspath('.'))
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from baselines.design2code.utils_ import extract_text_from_html, html2shot, image2md5, parse_content
from utils.log import logger, init_logger
from my_datasets import *
from vendors.google__ import gemini
from vendors.openai__ import gpt4o
from vendors.deepseek__ import deepseek_vl2, init_llm

SEED = 2026

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
    # agents.SEED = SEED
    
set_global_seed(SEED)

def direct_prompting(image):
    '''
    {original input image + prompt} -> {output html}
    '''

    ## the prompt 
    direct_prompt = ""
    direct_prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
    direct_prompt += "A user will provide you with a screenshot of a webpage.\n"
    direct_prompt += "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
    direct_prompt += "Include all CSS code in the HTML file itself.\n"
    direct_prompt += "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
    direct_prompt += "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
    direct_prompt += "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n"
    direct_prompt += "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
    direct_prompt += "Respond with the content of the HTML+CSS file:\n"
    
    ## call 
    html = BACKBONE(direct_prompt,[image],temperature=0.0,seed=SEED)
    return parse_content(html)

def text_augmented_prompting(image, html_content):
    '''
    {original input image + extracted text + prompt} -> {output html}
    '''

    ## extract all texts from the webpage 
    texts = "\n".join(extract_text_from_html(html_content))

    ## the prompt
    text_augmented_prompt = ""
    text_augmented_prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
    text_augmented_prompt += "A user will provide you with a screenshot of a webpage, along with all texts that they want to put on the webpage.\n"
    text_augmented_prompt += "The text elements are:\n" + texts + "\n"
    text_augmented_prompt += "You should generate the correct layout structure for the webpage, and put the texts in the correct places so that the resultant webpage will look the same as the given one.\n"
    text_augmented_prompt += "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
    text_augmented_prompt += "Include all CSS code in the HTML file itself.\n"
    text_augmented_prompt += "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
    text_augmented_prompt += "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
    text_augmented_prompt += "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n"
    text_augmented_prompt += "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
    text_augmented_prompt += "Respond with the content of the HTML+CSS file (directly start with the code, do not add any additional explanation):\n"

    ## call
    html = BACKBONE(text_augmented_prompt,[image],temperature=0.0,seed=SEED)

    return parse_content(html)

def visual_revision_prompting(input_image,html_content, orig_image,orig_html):
    '''
    {input image + initial output image + initial output html + oracle extracted text} -> {revised output html}
    '''

    texts = "\n".join(extract_text_from_html(html_content))

    prompt = ""
    prompt += "You are an expert web developer who specializes in HTML and CSS.\n"
    prompt += "I have an HTML file for implementing a webpage but it has some missing or wrong elements that are different from the original webpage. The current implementation I have is:\n" + orig_html + "\n\n"
    prompt += "I will provide the reference webpage that I want to build as well as the rendered webpage of the current implementation.\n"
    prompt += "I also provide you all the texts that I want to include in the webpage here:\n"
    prompt += "\n".join(texts) + "\n\n"
    prompt += "Please compare the two webpages and refer to the provided text elements to be included, and revise the original HTML implementation to make it look exactly like the reference webpage. Make sure the code is syntactically correct and can render into a well-formed webpage. You can use \"rick.jpg\" as the placeholder image file.\n"
    prompt += "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
    prompt += "Respond directly with the content of the new revised and improved HTML file without any extra explanations:\n"

    text = "the first image is reference webpage,and the second image is current webpage"
    html = BACKBONE(prompt,[text,input_image, orig_image],temperature=0.0,seed=SEED)
    return parse_content(html)

def save_result(ans_image,ans_html,pre_image,pre_html,path):
    ans_image.save(str(path/'answer.png'))
    pre_image.save(str(path/"prediction.png"))

    with open(path/'answer.html', 'w') as f:
        f.write(ans_html)

    with open(path/'prediction.html', 'w') as f:
        f.write(pre_html) 

METHODS = [direct_prompting,text_augmented_prompting,visual_revision_prompting]
BACKBONES = [gpt4o, gemini,deepseek_vl2]
DATASETS = [D2CHardDataset,V2UDataset]
BACKBONE = None

def main(dataset, medthod_id, out_dir,origin_output=None):
    for i, item in tqdm(enumerate(dataset)):

        md5 = image2md5(item['image'])
        out_dir = save_dir / md5
        out_dir.mkdir(exist_ok=True, parents= True)
        html = None
        logger.info(f"path: {str(out_dir)}")
        try:
            if medthod_id == 0:
                html = direct_prompting(item['image'])
            elif medthod_id == 1:
                html = text_augmented_prompting(item['image'],item['text'])     
            elif medthod_id == 2:
                if origin_output is None:
                    logger.error("")
                    break
                # 
                orig_path = Path(origin_output) / md5
                orig_html_path = str(orig_path / "prediction.html")
                if not os.path.exists(orig_html_path):
                    logger.info(f" {orig_html_path} ，")
                    # continue
                with open(orig_html_path) as f:
                    orig_html = f.read()
                
                orig_image_path = str(orig_path / "prediction.png")
                if not os.path.exists(orig_image_path):
                    logger.info(f" {orig_image_path} ，")
                    # continue
                orig_image = Image.open(orig_image_path)
                html = visual_revision_prompting(item["image"], item['text'],orig_image, orig_html)
        except Exception as e:
            exception_info = traceback.format_exc()
            logger.error(f"EEORR:{e}\nTrace:\n{exception_info}")
            continue
        # ,html
        if html is not None:
            save_result(item['image'],item['text'],html2shot(html_content=html),html,out_dir)
            logger.info(f"index = {i} ")
        else:
            logger.info(f"html  index = {i} ")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset','-ds',type=int,default=0,choices=list(range(len(DATASETS))))
    parser.add_argument('--method','-m', type=int, default=0,choices=list(range(len(METHODS))))
    parser.add_argument('--output', '-o', type=str, default='./output')
    parser.add_argument('--origin_output', '-op', type=str,default=None)
    parser.add_argument('--backbone','-b',type=int,default=0,choices=list(range(len(BACKBONES))))
    args = parser.parse_args()
    
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    save_dir=Path(args.output) / f"{time_str}"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    init_logger(logger, logfile=str(save_dir/'log.txt'))  
    logger.info(args)

    if args.backbone ==2: # deepseek-vl2 via vllm
        init_llm(torch.cuda.device_count(), SEED)

    # backbond
    BACKBONE = BACKBONES[args.backbone]
  
    # 
    ds = DATASETS[args.dataset]()
    main(ds, args.method, save_dir, args.origin_output)
    

