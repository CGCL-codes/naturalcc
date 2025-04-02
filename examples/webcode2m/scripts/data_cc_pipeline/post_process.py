from transformers import pipeline
from bs4 import BeautifulSoup

PIPE = None
def detect_lang(html_list, max_len=4096,device="cpu"):
    text_list=[]
    for html in html_list:
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        text=text[: min(max_len, len(text))]
        text_list.append(text)
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    global PIPE
    if not PIPE:
        PIPE = pipeline("text-classification", model=model_ckpt, device=device)
    res = PIPE(text_list, top_k=1, truncation=True)
    return res

