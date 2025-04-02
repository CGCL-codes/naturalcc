import gradio as gr
import os
os.environ['https_proxy'] = '127.0.0.1:27890'
os.environ['https_proxy'] = '127.0.0.1:27890'
from PIL import Image
from tqdm import tqdm
import asyncio
from io import BytesIO
from PIL import Image
from playwright.async_api import async_playwright
import json
import random
loop = asyncio.get_event_loop()

async def init_browser():
    p = await async_playwright().start()
    return await p.chromium.launch()

browser = loop.run_until_complete(init_browser())
page = loop.run_until_complete(browser.new_page())

async def gen_shortcut(html):
    global browser, page
    await page.set_viewport_size({"width": 640, "height": 360})
    await page.set_content(html, wait_until='networkidle')  # 确保所有资源加载完成

    # 获取页面尺寸
    body = await page.query_selector('body')
    body_width = await body.evaluate('(body) => body.scrollWidth')
    body_height = await body.evaluate('(body) => body.scrollHeight')
    await page.set_viewport_size({"width": body_width, "height": body_height})

    # 截图
    screenshot_bytes = await page.screenshot()
    image = Image.open(BytesIO(screenshot_bytes))

    return image

data_dir = "/data02/projects/vision2ui/results"
len_types = ['long', 'mid', 'short']
save_freq = 5
anonymous = True

benchmarks = {}
bch_now = 'vision2ui'
item_idx = 0
data_now = []

for filename in os.listdir(data_dir):
    model, benchmark = '-'.join(filename.split('-')[:-1]), filename.split('-')[-1]
    if benchmark not in benchmarks:
        benchmarks[benchmark] = {
            'models': [],
            'data': {},
            'result': {}
        }
    if model not in benchmarks[benchmark]['models'] and (model in ['GPT4V', 'GPT4o', 'vu-agent-4v-opt'] or 'onevision' in model):
        benchmarks[benchmark]['models'].append(model)    

# 选择要人类评测的数据集
def pick_benmark(benchmark):
    global bch_now
    bch_now = benchmark
    # 选择测试集后，重建数据索引
    model_list = benchmarks[benchmark]['models']
    for model in tqdm(model_list):
        if benchmark == 'vision2ui':
            for len_type in len_types:
                search_dir = f'{data_dir}/{model}-{benchmark}/{len_type}'
                for hash in os.listdir(search_dir)[:3]:
                    if f'{len_type}-{hash}' not in benchmarks[benchmark]['data']:
                        benchmarks[benchmark]['data'][f'{len_type}-{hash}'] = {}
                    benchmarks[benchmark]['data'][f'{len_type}-{hash}'][model] = {
                        'path': f'{search_dir}/{hash}'
                    }
        else:
            search_dir = f'{data_dir}/{model}-{benchmark}'
            for hash in os.listdir(search_dir):
                if f'{hash}' not in benchmarks[benchmark]['data']:
                    benchmarks[benchmark]['data'][hash] = {}
                benchmarks[benchmark]['data'][hash][model] = {
                    'path': f'{search_dir}/{hash}'
                }
    return pick_item(0)

# 自定义排序策略
def order_item(item):
    global data_now
    data_now = []
    # 随机排序策略
    keys = list(item.keys())
    random.shuffle(keys)
    for key in keys:
        if not isinstance(item[key], str) and item[key]['data']['image'][0]:
            data_now.append(item[key]['data']['image'])

    data = []
    for idx, x in enumerate(data_now):
        data.append((x[0], f"Img{idx+1}{'' if anonymous else ' ('+x[1]+')'}"))
    return data

# 渲染html生成图片
def gen_data(model, item):
    try:
        with open(f"{item[model]['path']}/prediction.html", 'r') as f:
            html = f.read()
        image = loop.run_until_complete(gen_shortcut(html))
    except Exception as e:
        print(e)
        image = None
    return model, {
        'html': html,
        'image': (image, model)
    }

# 选择一条数据
def pick_item(idx):
    global item_idx, bch_now, benchmarks
    hash_list = sorted(list(benchmarks[bch_now]['data'].keys()))

    if idx > len(hash_list)-1:
        item = benchmarks[bch_now]['data'][hash_list[item_idx]]
        return item['refer_image'], order_item(item), f'{item_idx+1}/{len(hash_list)} (Finished)'
    if idx < 0:
        item = benchmarks[bch_now]['data'][hash_list[item_idx]]
        return item['refer_image'], order_item(item), f'{item_idx+1}/{len(hash_list)}', ''
    
    print(hash_list[item_idx])
    
    item_idx = idx
    item = benchmarks[bch_now]['data'][hash_list[idx]]
    if 'refer_image' not in item:
        for model in tqdm(item.keys()):
            model, result = gen_data(model, item)
            item[model]['data'] = result
        item['refer_image'] = f"{item[list(item.keys())[0]]['path']}/image.png"

    return item['refer_image'], order_item(item), f'{idx+1}/{len(hash_list)}', ""
def pick_last():
    global item_idx
    return pick_item(item_idx-1)
def pick_next(score_input, save_filename):
    global item_idx
    hash_list = sorted(list(benchmarks[bch_now]['data'].keys()))
    item = benchmarks[bch_now]['data'][hash_list[item_idx]]

    if score_input == "Finished":
        return item['refer_image'], order_item(item), f'{item_idx+1}/{len(hash_list)} (Finished)', "Finished"

    try:
        score(score_input)
    except Exception as e:
        return item['refer_image'], order_item(item), f'{item_idx+1}/{len(hash_list)}', e
    
    if len(benchmarks[bch_now]['result'].keys()) % save_freq == 0 or item_idx == len(hash_list)-1:
        save_result(save_filename)

    if item_idx < len(hash_list)-1:
        return pick_item(item_idx+1)
    else:
        return item['refer_image'], order_item(item), f'{item_idx+1}/{len(hash_list)} (Finished)', "Finished"

# 打标
def tag(option, save_filename):
    global benchmarks, bch_now, item_idx, data_now, save_freq
    answer_idx = int(option[-1])-1
    if answer_idx > len(data_now)-1:
        save_result(save_filename)
        return pick_item(item_idx)
    benchmarks[bch_now]['result'][item_idx] = data_now[answer_idx][1]
    if len(benchmarks[bch_now]['result'].keys()) % save_freq == 0:
        save_result(save_filename)
    return pick_next()

# 打标2 - 打分
def score(scores):
    global benchmarks, bch_now, item_idx, data_now, save_freq
    hash_list = sorted(list(benchmarks[bch_now]['data'].keys()))
    for i in range(3):
        check_str = str(i+1)
        if check_str not in scores:
            raise ValueError(f'Score {i+1} not in input')
    group_score = {}
    for idx, item in enumerate(data_now):
        group_score[item[1]] = scores[idx]
    benchmarks[bch_now]['result'][hash_list[item_idx]] = scores

# 保存结果
def save_result(filename):
    global bch_now, benchmarks
    result = benchmarks[bch_now]['result']
    # count = dict(Counter(result.values()))
    # print(bch_now, count)
    with open(f'{bch_now}_{filename}.json','w') as f:
        json.dump(result, f, indent=2)

# 设置自动保存的频率
def set_auto_save_freq(freq):
    global save_freq
    save_freq = freq

# 设置是否匿名选择
def set_anonymous(value):
    global anonymous
    anonymous = value

with gr.Blocks() as demo:
    with gr.Row():
        benchmark_choices = gr.Dropdown(choices=benchmarks, value="-1", label="Benchmarks")
        save_freq_picker = gr.Number(label="Auto save freq", value=save_freq, minimum=1, maximum=100)
        anonymous_switch = gr.Checkbox(label="Anonymous", value=True)
        save_filename = gr.Text(label="Save filename", value="test", interactive=True)

    with gr.Blocks():
        with gr.Row():
            with gr.Column(scale=1):
                refer_image = gr.Image(label="Refer image", height=500)
                index = gr.Text(label="Progress", interactive=False)
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Redered results", columns=4, rows=1, object_fit="contain", height=490
                )
                with gr.Row():
                    score_input = gr.Text(label="Score", interactive=True)
                    gr.Markdown("将1|2|3分 分别分配给一张图\n\n最好的给3分,第二给2分,最次给1分\n\n输入中分数无需分隔,例`132`")
                    # btn1 = gr.Button("Img1", variant="primary")
                    # btn2 = gr.Button("Img2", variant="primary")
                    # btn3 = gr.Button("Img3", variant="primary")
                    # btn4 = gr.Button("Img4", variant="primary")
                    # btn5 = gr.Button("Img5", variant="primary")
                    # btn6 = gr.Button("Img6", variant="primary")
                    # btn7 = gr.Button("Img7", variant="primary")
                    # btn8 = gr.Button("Img8", variant="primary")
                    # btn9 = gr.Button("Img9", variant="primary")
                    # btn10 = gr.Button("Img10", variant="primary")
        
        with gr.Row():
            btn_last = gr.Button("Last group")
            btn_next = gr.Button("Next group")


    benchmark_choices.change(pick_benmark, inputs=[benchmark_choices], outputs=[refer_image, gallery, index])
    btn_last.click(pick_last, outputs=[refer_image, gallery, index, score_input])
    btn_next.click(pick_next, inputs=[score_input, save_filename], outputs=[refer_image, gallery, index, score_input])
    # btn1.click(tag, inputs=[btn1, save_filename], outputs=[refer_image, gallery, index])
    # btn2.click(tag, inputs=[btn2, save_filename], outputs=[refer_image, gallery, index])
    # btn3.click(tag, inputs=[btn3, save_filename], outputs=[refer_image, gallery, index])
    # btn4.click(tag, inputs=[btn4, save_filename], outputs=[refer_image, gallery, index])
    # btn5.click(tag, inputs=[btn5, save_filename], outputs=[refer_image, gallery, index])
    # btn6.click(tag, inputs=[btn6, save_filename], outputs=[refer_image, gallery, index])
    # btn7.click(tag, inputs=[btn7, save_filename], outputs=[refer_image, gallery, index])
    # btn8.click(tag, inputs=[btn8, save_filename], outputs=[refer_image, gallery, index])
    # btn9.click(tag, inputs=[btn9, save_filename], outputs=[refer_image, gallery, index])
    # btn10.click(tag, inputs=[btn10, save_filename], outputs=[refer_image, gallery, index])
    save_freq_picker.change(set_auto_save_freq, inputs=[save_freq_picker])
    anonymous_switch.change(set_anonymous, inputs=[anonymous_switch])


if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=10009)
    # with open('/data02/users/lz/code/tmp/test.html','r') as f:
    #     html = f.read()
    # img = loop.run_until_complete(gen_shortcut(html))
    # img.save('/data02/users/lz/code/tmp/scshot.png')