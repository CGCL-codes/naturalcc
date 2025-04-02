import gradio as gr
import datasets
from PIL import Image
import io
import os

# 加载数据集
ds_path='/data02/starmage/datasets/cc/bench/long.parquet'
ds_save_path= '/data02/starmage/datasets/cc/bench/long_good.parquet'
dataset = datasets.Dataset.from_parquet(ds_path)  # 请替换成实际的数据集名称
all_ids = list(range(len(dataset)))
max_height = 600

def resize(im):
    return im
    ra = max_height /im.size[1] 
    return im.resize((int(im.size[0]*ra), int(im.size[1]*ra)))

# 数据点索引
cur_index = 0
id_list=[]

def next():
    global cur_index    
    #id_list.append(cur_index)  # 根据你的数据集字段调整
    cur_index = (cur_index+1+len(dataset)) % len(dataset)    

    return dataset[cur_index]['image'], f'[{cur_index}|{len(dataset)}]: {len(set(id_list))}\n {list(set(id_list))}', False

def drop():
    global cur_index    
    id_list.append(cur_index)  # 根据你的数据集字段调整
    return next()
    
def save_ids():
    ds_good = dataset.select(list((set(all_ids) - set(id_list)))).to_parquet(ds_save_path)
    return "ID列表已保存到文件"


with gr.Blocks() as demo:
    with gr.Row():
        image = gr.Image(label="Image",type="pil", value=dataset[cur_index]['image'], format='PNG')
        with gr.Row():            
            text = gr.Text(label="DropCount", value=f'[{cur_index}|{len(dataset)}]')
            drop_btn=gr.Button("Drop")
            drop_btn.click(fn=drop, outputs=[image, text])
            next_btn=gr.Button("next")
            next_btn.click(fn=next, outputs=[image, text])
            save_btn=gr.Button("save")
            save_btn.click(fn=save_ids)        

demo.launch()