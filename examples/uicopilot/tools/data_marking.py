import gradio as gr
import os
import json
from PIL import Image
from html_utils import gen_shortcut

data_dir = '/data02/users/lz/code/UICoder/datasets/c4/format_html'
volumes = sorted(os.listdir(data_dir),key=lambda x:int(x))
index = 0
batch_size = 4

result = []
        
async def reset():
    global index,result
    index = 0
    result = []
    return (await next())
    
async def next():
    global index,batch_size,volumes
    images = []
    for i in range(batch_size):
        if index+i > len(volumes)-1:
            break
        images.append((os.path.join(data_dir,volumes[index+i],'image.png'),f'{volumes[index+i]}'))
        
    if index < len(volumes):
        index += batch_size
    return images

async def mark(i2s):
    global index,volumes
    i2s = list(map(lambda x:int(x[-1])-1,i2s))
    for i in range(batch_size):
        result.append({
            'volume': index+i-batch_size,
            'label': 'good' if (i in i2s) else 'bad'
        })
    with open('./marking_result.json','w') as f:
        json.dump(result,f)
    return (await next())


with gr.Blocks() as demo:
    gallery = gr.Gallery(
        label="HTML rendered images", show_label=False, columns=4, rows=batch_size/4, object_fit="contain", height="500"
    )
    btn = gr.Button("reset")
    btn1 = gr.Button("next batch")
    checkBoxGroup = gr.CheckboxGroup(['P1','P2','P3','P4'],label='Choose good shortcuts')
    btn2 = gr.Button("mark")
    btn.click(reset, None, gallery)
    btn1.click(next, None, gallery)
    btn2.click(mark, checkBoxGroup, gallery)

if __name__ == '__main__':
    demo.launch()