import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
sys.path.append(os.path.abspath('.'))
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
import torch
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import argparse
from pathlib import Path
from metrics import *
from evaluation.design2code.visual_score import visual_score_v3,pre_process
import cv2
import pandas as pd
import datetime
from PIL import Image
import time
from utils.processor import MultiProcessor    
import multiprocessing
import signal
from evaluation.mrweb.emd_similarity import emd_similarity
from evaluation.mrweb.study import LPIPSScorer, psnr_score, mae_score
import traceback
import glob

from utils.log import logger, init_logger

SEED = 1037

def html_sim_scores(html1_path, html2_path):   
    with open(html1_path, "r") as f:
         html1 = f.read()
    with open(html2_path, "r") as f:
         html2 = f.read()
    assert len(html1) >0 and len(html2)>0, "The html must not be empty!"
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
        logger.info(f"cp {str(imgs)} {str(pred_html.parent)}/")
        try:    
            pre_process(str(pred_html))
        except Exception as e:
            logger.info(f"fail to prreprocess: {e}")
            continue            
        answer_html=input_dir/f'{file}/answer.html'
        pred_screenshot=output_dir/f'{file}/prediction.png'
        answer_screenshot=output_dir/f'{file}/answer.png'
        os.system(f"python scripts/evaluation/html2screenshot.py --input {str(answer_html)} --output {str(answer_screenshot)}")
        os.system(f"python scripts/evaluation/html2screenshot.py --input {str(pred_html)} --output {str(pred_screenshot)}")
        yield pred_html,pred_screenshot,answer_html,answer_screenshot

def genertor1(input_dir:Path, output_dir:Path):
    preds_html_dir = input_dir / "preds/html"
    preds_html_dir.mkdir(exist_ok=True, parents=True)
    preds_screenshot_dir = input_dir / "preds/screenshot"
    preds_screenshot_dir.mkdir(exist_ok=True, parents=True)
    answers_screenshot_dir = input_dir / "answers/screenshot"
    answers_screenshot_dir.mkdir(exist_ok=True, parents=True)
    answers_html_dir = input_dir / "answers/html"
    answers_html_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Taking screenshot of origin htmls ...")
    os.system(f"python scripts/evaluation/html2screenshot.py --input {str(answers_html_dir)} --output {str(answers_screenshot_dir)}")
    logger.info("Taking screenshot of predictions ...")
    os.system(f"python scripts/evaluation/html2screenshot.py --input {str(preds_html_dir)} --output {str(preds_screenshot_dir)}")
    for file in tqdm(os.listdir(preds_html_dir)):
        pred_html = preds_html_dir/ file
        pred_screenshot = preds_screenshot_dir/ f"{file.split('.')[0]}.png"
        answer_html = answers_html_dir/ file
        answer_screenshot = answers_screenshot_dir/ f"{file.split('.')[0]}.png"
        yield pred_html,pred_screenshot,answer_html,answer_screenshot
        
# agent，sampleagent            
def genertor2(input_dir:Path, output_dir:Path):
    for file in os.listdir(input_dir):
        pred_html = input_dir/f'{file}/prediction.html'
        # pred_html = input_dir/f'{file}/assemble_3.html'
        answer_html=input_dir/f'{file}/answer.html'
        pred_screenshot=input_dir/f'{file}/prediction.png'
        # pred_screenshot=input_dir/f'{file}/assemble_3.png'
        answer_screenshot=input_dir/f'{file}/answer.png'

        # ，
        if not all(os.path.exists(str(path)) for path in [pred_html, answer_html, pred_screenshot, answer_screenshot]):
            logger.info(f"Skipping {file}, one or more paths do not exist.")
            continue

        """
        try:    
            pre_process(str(pred_html))
        except Exception as e:
            logger.info(f"fail to prreprocess: {e}")
            continue        
        """   
        yield pred_html,pred_screenshot,answer_html,answer_screenshot
        
def get_files_scores(dir):
    print(dir)
    pattern1 = os.path.join(dir, 'assemble_abs_0*.png') #assemble_abs_0.8283108813529717.png
    pattern2 = os.path.join(dir, 'assemble_agent_0_0*.png')

    # 
    png_abs = glob.glob(pattern1)[0]
    png_agent = glob.glob(pattern2)[0]

    html_abs = png_abs[:png_abs.rfind('_')]+'.html'
    html_agent = png_agent[:png_agent.rfind('_')]+'.html'

    score_abs = float(Path(png_abs).name[:-4].split('_')[-1])
    score_agent = float(Path(png_agent).name[:-4].split('_')[-1])

    return [[score_abs, png_abs, html_abs],
        [score_agent, png_agent, html_agent]]


#         
def genertor3(input_dir:Path, output_dir:Path):
    for file in os.listdir(input_dir):
        try: 
            res = get_files_scores(str(input_dir/file))
        except:
            logger.error(f"Fail to get files and scores: {str(input_dir/file)}")
            continue
        answer_html=input_dir/f'{file}/answer.html'
        answer_screenshot=input_dir/f'{file}/answer.png'
        pred_html = Path(res[0][2])
        pred_screenshot= Path(res[0][1])        

        # ，
        if not all(os.path.exists(str(path)) for path in [pred_html, answer_html, pred_screenshot, answer_screenshot]):
            logger.info(f"Skipping {file}, one or more paths do not exist.")
            continue

        """
        try:    
            pre_process(str(pred_html))
        except Exception as e:
            logger.info(f"fail to prreprocess: {e}")
            continue        
        """   
        yield pred_html,pred_screenshot,answer_html,answer_screenshot

# agent        
def genertor4(input_dir:Path, output_dir:Path):
    for file in os.listdir(input_dir):
        try:
            res = get_files_scores(str(input_dir/file))
        except:
            logger.error(f"Fail to get files and scores: {str(input_dir/file)}")
            continue
        answer_html=input_dir/f'{file}/answer.html'
        answer_screenshot=input_dir/f'{file}/answer.png'
        best_id = 0 if res[0][0] > res[1][0] else 1
        pred_html = Path(res[best_id][2])
        pred_screenshot=Path(res[best_id][1])        

        # ，
        if not all(os.path.exists(str(path)) for path in [pred_html, answer_html, pred_screenshot, answer_screenshot]):
            logger.info(f"Skipping {file}, one or more paths do not exist.")
            continue

        """
        try:    
            pre_process(str(pred_html))
        except Exception as e:
            logger.info(f"fail to prreprocess: {e}")
            continue        
        """   
        yield pred_html,pred_screenshot,answer_html,answer_screenshot




def eval_work(data, out_df):
    pred_html,pred_screenshot,answer_html,answer_screenshot = data
    if not pred_screenshot.exists() or not answer_screenshot.exists():
        logger.info(f"Screenshot file not exits:\n {str(pred_screenshot)}.\n{str(answer_screenshot)}.")
        return
    bleu, rouge, tree_bleu, tree_rouge_1 = html_sim_scores(answer_html, pred_html)
    mse_value, ssim_value, clip_sim = image_sim_scores(str(pred_screenshot), str(answer_screenshot))
    
    im_answer_screenshot = Image.open(answer_screenshot)
    im_pred_screenshot = Image.open(pred_screenshot)

 
    psnr_value = psnr_score(im_answer_screenshot, im_pred_screenshot)
    mae_value = mae_score(im_answer_screenshot, im_pred_screenshot)
    emd_score = emd_similarity(im_answer_screenshot, im_pred_screenshot, max_size=96, mode="RGB")

    tmp_dir:Path = pred_screenshot.parent / 'tmp'
    tmp_dir.mkdir(exist_ok=True, parents=True)
    _, _, block_match, text_match, position_match, text_color_match, clip_score = \
        visual_score_v3(str(answer_html), str(pred_html), str(answer_screenshot), str(pred_screenshot), str(tmp_dir), device="cpu")
    
    with open(out_df, "a+") as f_csv:
        f_csv.write(f"{str(pred_html)},{str(answer_html)},{bleu},{rouge},{tree_bleu}, {tree_rouge_1},{mse_value},{ssim_value},{clip_sim},\
                    {psnr_value},{mae_value},{emd_score}, \
            {block_match},{text_match},{position_match},{text_color_match},{clip_score}\n")

def eval(input_dir, output_dir, generator_choice):
    generator_map={'0':genertor0,'1':genertor1,'2':genertor2,'3':genertor3,'4':genertor4}
    generator=generator_map[generator_choice]
    device = 'cuda'
    torch.manual_seed(SEED)    
    
    # take screenshots, the playwright has to work in the main process.    
    # caculate all the metrics
    input_dir = Path(input_dir)
    """
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d%H%M%S")
    output_dir = Path(output_dir) / f"eval_{input_dir.name}_{time_string}"
    """

    logger.info("Caculating metrics:")
    out_df = os.path.join(output_dir, "metrics_result.csv")
    # block_match, text_match, position_match, text_color_match, clip_score,\
    with open(out_df, "w") as f_csv:
        f_csv.write("origin,pred,bleu,rouge,tree_bleu,tree_rouge_1, mse_value,ssim_value, clip_sim,\
                    psnr_value,mae_value,emd_score,\
                    block_match, text_match, position_match, text_color_match, clip_score\n")
    tbar = tqdm(total=len(os.listdir(input_dir))) 
    def cb(res):
        tbar.update(1)   
    #pool =  MultiProcessor(12)   
    for i, data in enumerate(generator(input_dir,output_dir)):
        logger.info(f"Evaluating {i}: {data[0]} ...")
        #pool.add_task(eval_work, (data,str(out_df)), cb)
        try:
            eval_work(data,str(out_df))
        except Exception as e:
            exception_info = traceback.format_exc()
            logger.error(f"EEORR:{e}\nTrace:\n{exception_info}")
        tbar.update(1)
    #pool.shutdown()
    df = pd.read_csv(out_df)
    for c in df.columns:
        if c not in ["origin","pred"]:
            logger.info(f"{c}:{df[c].mean():.4f}")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process two path strings.')
    # Define the arguments
    parser.add_argument('--input', "-i", type=str, default='output/2025-01-09_00-38-49_926300')  
    parser.add_argument('--output', "-o", type=str, default='output/eval/4o-agent') 
    parser.add_argument('--generator', "-g", type=str, choices=['0','1','2','3', '4'], default='2') 
    # Parse the arguments
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_logger(logger, logfile=str(output_dir/'log.txt'))  
    logger.info(args)
    
    """
    def signal_handler(signal, frame):
        logger.info(f'signal {signal} recieved, exit.')         
        for p in multiprocessing.active_children():            
            # 
            # 
            os.kill(p.pid, signal.SIGKILL)
            # 
        os._exit(1)    
            # 
    signal.signal(signal.SIGINT, signal_handler)  
    """ 
    
    eval(args.input, args.output,args.generator)