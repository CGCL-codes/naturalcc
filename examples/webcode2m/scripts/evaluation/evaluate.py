import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
sys.path.append(os.path.abspath('.'))
import torch
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import argparse
from pathlib import Path
from metrics import *
from scripts.evaluation.design2code.visual_score import visual_score_v3,pre_process
import cv2
import pandas as pd
import datetime
from scripts.train.vars import SEED
from PIL import Image
import time
from tools.processor import MultiProcessor    
import multiprocessing
import signal

def html_sim_scores(html1_path, html2_path):   
    with open(html1_path, "r") as f:
         html1 = f.read()
    with open(html2_path, "r") as f:
         html2 = f.read()
    sys.setrecursionlimit(6000)
    bleu, rouge = bleu_rouge(html1, html2)
    tree_bleu, tree_rouge_1 = dom_sim(html1, html2)
 
    return (bleu, rouge, tree_bleu, tree_rouge_1)

def image_sim_scores(img1_path, img2_path):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(
        img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4
    )    

    mse_value = mse(img1, img2)
    ssim_value = ssim(img1, img2)
    clip_sim_value = clip_sim(Image.open(img1_path), Image.open(img2_path), 'cpu')

    return mse_value, ssim_value,clip_sim_value

def genertor0(input_dir:Path, output_dir:Path):
    for file in os.listdir(input_dir):
        pred_html_origin=input_dir/f'{file}/prediction.html'
        if not pred_html_origin.exists():
            pred_html_origin=input_dir/f'{file}/pred.html'
        pred_html = output_dir/f'{file}/prediction.html'
        imgs = input_dir/f'{file}/*.png'
        pred_html.parent.mkdir(exist_ok=True, parents=True)
        os.system(f'cp {str(imgs)} {str(pred_html.parent)}/') 
        os.system(f'cp {str(pred_html_origin)} {str(pred_html)}')   
        print(f"cp {str(imgs)} {str(pred_html.parent)}/")
        try:    
            pre_process(str(pred_html))
        except Exception as e:
            print(f"fail to prreprocess: {e}")
            continue            
        answer_html=input_dir/f'{file}/answer.html'
        pred_screenshot=output_dir/f'{file}/prediction.png'
        answer_screenshot=output_dir/f'{file}/answer.png'
        os.system(f"python scripts/evaluation/html2screenshot.py --input {str(answer_html)} --output {str(answer_screenshot)}")
        os.system(f"python scripts/evaluation/html2screenshot.py --input {str(pred_html)} --output {str(pred_screenshot)}")
        yield pred_html,pred_screenshot,answer_html,answer_screenshot

def genertor1(input_dir, output_dir:Path):
    preds_html_dir = input_dir / "preds/html"
    preds_html_dir.mkdir(exist_ok=True, parents=True)
    preds_screenshot_dir = input_dir / "preds/screenshot"
    preds_screenshot_dir.mkdir(exist_ok=True, parents=True)
    answers_screenshot_dir = input_dir / "answers/screenshot"
    answers_screenshot_dir.mkdir(exist_ok=True, parents=True)
    answers_html_dir = input_dir / "answers/html"
    answers_html_dir.mkdir(exist_ok=True, parents=True)
    print("Taking screenshot of origin htmls ...")
    os.system(f"python scripts/evaluation/html2screenshot.py --input {str(answers_html_dir)} --output {str(answers_screenshot_dir)}")
    print("Taking screenshot of predictions ...")
    os.system(f"python scripts/evaluation/html2screenshot.py --input {str(preds_html_dir)} --output {str(preds_screenshot_dir)}")
    for file in tqdm(os.listdir(preds_html_dir)):
        pred_html = preds_html_dir/ file
        pred_screenshot = preds_screenshot_dir/ f"{file.split('.')[0]}.png"
        answer_html = answers_html_dir/ file
        answer_screenshot = answers_screenshot_dir/ f"{file.split('.')[0]}.png"
        yield pred_html,pred_screenshot,answer_html,answer_screenshot

def eval_work(data, out_df):
    pred_html,pred_screenshot,answer_html,answer_screenshot = data
    if not pred_screenshot.exists() or not answer_screenshot.exists():
        print(f"Screenshot file not exits:\n {str(pred_screenshot)}.\n{str(answer_screenshot)}.")
        return
    bleu, rouge, tree_bleu, tree_rouge_1 = html_sim_scores(answer_html, pred_html)
    mse_value, ssim_value, clip_sim = image_sim_scores(str(pred_screenshot), str(answer_screenshot))

    _, _, block_match, text_match, position_match, text_color_match, clip_score = \
        visual_score_v3(str(answer_html), str(pred_html), str(answer_screenshot), str(pred_screenshot), str(pred_screenshot.parent), device="cpu")
    with open(out_df, "a+") as f_csv:
        f_csv.write(f"{str(pred_html)},{str(answer_html)},{bleu},{rouge},{tree_bleu}, {tree_rouge_1},{mse_value},{ssim_value},{clip_sim},\
                    {block_match}, {text_match}, {position_match}, {text_color_match}, {clip_score}\n")
        #    {block_match},{text_match},{position_match},{text_color_match},{clip_score}\n")

def eval(input_dir, output_dir, generator_choice):
    generator_map={'0':genertor0,'1':genertor1}
    generator=generator_map[generator_choice]
    device = 'cuda'
    torch.manual_seed(SEED)    
    
    # take screenshots, the playwright has to work in the main process.    
    # caculate all the metrics
    input_dir = Path(input_dir)
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d%H%M%S")
    output_dir = Path(output_dir) / f"eval_{input_dir.name}_{time_string}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Caculating metrics:")
    out_df = output_dir / "metrics_result.csv"
    with open(out_df, "w") as f_csv:
        f_csv.write("origin,pred,bleu,rouge,tree_bleu,tree_rouge_1, mse_value,ssim_value, clip_sim, block_match, text_match, position_match, text_color_match, clip_score\n")
    tbar = tqdm(total=len(os.listdir(input_dir))) 
    def cb(res):
        tbar.update(1)   
    pool =  MultiProcessor(12)   
    for data in generator(input_dir,output_dir):
        pool.add_task(eval_work, (data,str(out_df)), cb)
    pool.shutdown()
    df = pd.read_csv(out_df)
    for c in df.columns:
        if c not in ["origin","pred"]:
            print(f"{c}:{df[c].mean():.4f}")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process two path strings.')
    # Define the arguments
    parser.add_argument('--input', "-i", type=str)  
    parser.add_argument('--output', "-o", type=str) 
    parser.add_argument('--generator', "-g", type=str, choices=['0','1'], default='0') 
    # Parse the arguments
    args = parser.parse_args()
    def signal_handler(signal, frame):
        print(f'signal {signal} recieved, exit.')         
        for p in multiprocessing.active_children():            
            # 获取堆栈信息并写入文件
            # 杀死子进程
            os.kill(p.pid, signal.SIGKILL)
            # 退出主进程
        os._exit(1)    
            # 设置信号处理程序
    signal.signal(signal.SIGINT, signal_handler)   
    print(args)
    eval(args.input, args.output,args.generator)