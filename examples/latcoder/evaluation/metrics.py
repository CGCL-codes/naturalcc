from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method4
from rouge import Rouge
from PIL import Image
from skimage.metrics import structural_similarity
import numpy as np
from bs4 import BeautifulSoup
from evaluation.html_tree import * 
import clip
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

CLIP_MODEL, CLIP_PREPROCESS = None,None
def clip_encode(ims,device='cuda'):
    global CLIP_MODEL
    global CLIP_PREPROCESS
    if not CLIP_MODEL:        
        CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        img_tmps = torch.stack([CLIP_PREPROCESS(im) for im in ims]).to(device)
        img_feas = CLIP_MODEL.encode_image(img_tmps).cpu()
    return img_feas   

def clip_sim(im1, im2, device='cuda'):
    feas = clip_encode([im1 , im2], device)
    return torch.nn.functional.cosine_similarity(feas[0], feas[1], dim=0).item()

def bleu_rouge(original: str, generated: str):
    # html
    soup1 = BeautifulSoup(original, "html.parser")
    soup2 = BeautifulSoup(generated, "html.parser")
    original = soup1.get_text().split()
    generated = soup2.get_text().split()
    # BLEU, 1-gram
    bleu = bleu_score.sentence_bleu([original], generated, weights=(1.0, 0, 0, 0), smoothing_function=bleu_score.SmoothingFunction().method4)
    rouge = Rouge()
    # rouge-1 recall
    rouge_scores = rouge.get_scores(" ".join(generated), " ".join(original))

    return bleu, rouge_scores[0]["rouge-1"]["r"]


def ssim(img1: np.ndarray, img2: np.ndarray):
    """
    [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
        (2004). Image quality assessment: From error visibility to
        structural similarity. IEEE Transactions on Image Processing,
        13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        :DOI:`10.1109/TIP.2003.819861`
    """
    # img2 = img2.resize(img1.size, Image.LANCZOS)
    assert (
        img2.shape == img1.shape
    ), "to caculate the SSIM, two images should have the same shape."
    ssim_value = structural_similarity(
        img1,
        img2,
        multichannel=True,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        data_range=255.0,
    )
    return ssim_value


def mse(img1: np.ndarray, img2: np.ndarray):
    assert (
        img2.size == img1.size
    ), "To caculate the MSE, two images should have the same shape."
    im_1 = img1 / 255.0
    im_2 = img2 / 255.0
    err = np.sum((im_1.astype("float") - im_2.astype("float")) ** 2)
    err /= float(im_1.shape[0] * im_1.shape[1])
    return err

def dom_sim(ref:str, cand:str):
    """
    matching，[1]
    match_score = count(cand matched in ref)/count(subtrees number of ref)
    
    [1] Ren, Shuo et al. “CodeBLEU: a Method for Automatic Evaluation of Code Synthesis.” ArXiv abs/2009.10297 (2020): n. pag.
    """
    # DOM
    ref_tree_nodes = html2tree(ref)
    cand_tree_nodes = html2tree(cand)
    
    if len(ref_tree_nodes) == 0 or len(ref_tree_nodes) == 0:
        return 0,0
    
    def collect_all_subtrees(nodes, height=1): #2
        subtrees = []
        for node in nodes:
            if len(node.childs) == 0:
                continue
            names = [node.name.strip().lower()]
            for child in node.childs:
                names.append(child.name.strip().lower())            
            
            subtrees.append('_'.join(names))
        return subtrees
    
    ref_subtree_seqs = collect_all_subtrees(ref_tree_nodes)
    cand_subtree_seqs = collect_all_subtrees(cand_tree_nodes)
    
    match_count = 0
    for seq in set(cand_subtree_seqs):
        if seq in set(ref_subtree_seqs):
            match_count += 1
    
    tree_rouge_1 = match_count/len(set(ref_subtree_seqs))
    
    match_count = 0
    for seq in cand_subtree_seqs:
        if seq in set(ref_subtree_seqs):
            match_count += 1
        
    tree_bleu = match_count/len(cand_subtree_seqs)
    
    return tree_bleu, tree_rouge_1

    
    
    