import gradio as gr
from PIL import Image
from datasets import Dataset
import os

data_dir = '/data02/users/lz/code/UICoder/datasets/c4-wash/H128-2560_C128-4096_R2/c4-format'
save_path = '/data02/users/lz/code/UICoder/datasets/c4-wash/H128-2560_C128-4096_R2/c4-format-marked'
volumes = os.listdir(data_dir)
images = list(map(lambda x: {'image_path': os.path.join(data_dir,x,'image.png'),'options': [False,False,False,False,False]},volumes))[:1000]
index = -1

def go(delta):
    global index
    index += delta
    if index > len(images)-1:
        index = len(images)-1
    elif index < 0:
        index = 0
    return *images[index]['options'],f'{index+1}/{len(images)}',images[index]['image_path']

def last():
    return go(-1)

def next():
    return go(1)

def mark(o1,o2,o3,o4,o5):
    global index
    if index > -1:
        images[index]['options'] = [o1,o2,o3,o4,o5]
    return go(1)
    
def save(progress=gr.Progress()):
    global save_path
    def generator():
        for item in progress.tqdm(images,total=len(images)):
            yield {
                "image": Image.open(item['image_path']),
                "struct": item['options'][0],
                "style": item['options'][1],
                "margin": item['options'][2],
                "color": item['options'][3],
                "aesthetics": item['options'][4],
                "score": sum([int(i) for i in item['options']])
            }
    dataset = Dataset.from_generator(generator)
    dataset.save_to_disk(save_path)
    return 'Saved successfully'

with gr.Blocks() as demo:
    gr.Markdown(value='# VISION2UI Marking Tool\n\n\n')
    with gr.Row():
        with gr.Column():
            image = gr.Image(label='Screenshot',height=600)
        with gr.Column():
            with gr.Blocks():
                with gr.Row():
                    option1 = gr.Checkbox(label='Standard web page layout (presence of a layout, not merely a single vertical arrangement)')
                with gr.Row():
                    option2 = gr.Checkbox(label='Conventional web page style (elements such as lists, blocks exhibit styles)')
                with gr.Row():
                    option3 = gr.Checkbox(label='Absence of excessive blank styles')
                with gr.Row():
                    option4 = gr.Checkbox(label='Diverse color combinations')
                with gr.Row():
                    option5 = gr.Checkbox(label='Aesthetically appealing')

            with gr.Blocks():
                with gr.Row():
                    b1 = gr.Button(value="LAST",variant='primary')
                    show_index = gr.Textbox(label='INDEX')
                    b2 = gr.Button(value="NEXT",variant='primary')
                    b1.click(last,inputs=[],outputs=[option1,option2,option3,option4,option5,show_index,image])
                    b2.click(next,inputs=[],outputs=[option1,option2,option3,option4,option5,show_index,image])
                
                # progress = gr.Text(label='进度条')

                with gr.Row():
                    b3 = gr.Button(value="SAVE",variant='primary')
                    b4 = gr.Button(value="MARK",variant='primary')
                    b3.click(save)
                    b4.click(mark,inputs=[option1,option2,option3,option4,option5],outputs=[option1,option2,option3,option4,option5,show_index,image])
    

        
    demo.launch()
