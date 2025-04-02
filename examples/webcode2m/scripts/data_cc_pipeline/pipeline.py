import sys, os
sys.path.append(os.path.abspath('.'))
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = './.playwright'
from threading import Thread
import multiprocessing
import argparse
from tools.log import logger
from queue import Queue
from warc import *
from format_utils import *
import subprocess
import socket
import json
import queue
import signal
import shutil
from tools.processor import MulThreading, MultiProcessor

token2char = 5 
def BboxTree2Html(node,style=False):
    dom_type = node['type']
    childDoms = list(map(lambda node: BboxTree2Html(node,style),node['children']))
    if style:
        tree = f"<{dom_type} style='{node['style'] if 'style' in node else ''}'>{''.join(childDoms)}</{dom_type}>"
    else:  
        tree = f"<{dom_type} bbox={node['bbox']}>{''.join(childDoms)}</{dom_type}>"
    return tree

def format(path, html_range, css_range):    
    with open(path,'r') as f:
        html0 = f.read()
        html,css,URI = splitHtmlCss(html0)    
        if not html:
            return None
        html = formatHtml(html,URI)
        css = formatCss(css,html)
        if len(html) < html_range[0]*token2char or len(html) > html_range[1]*token2char:
            # raise ValueError(f'html length of {len(html)} not allowed')
            return None
            
        if len(css) < css_range[0]*token2char or len(css) > css_range[1]*token2char:
            # raise ValueError(f'css length of {len(html)} not allowed')
            return None

    html = mergeHtmlCss(html,css)
    
    return html

    
def download_task(chunk, volume, out_dir : Path):
    warc_table = get_warc_table(out_dir)    
    warc_dir = out_dir / "warc"
    warc_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Downloading chunk {chunk:03}, volume {volume:03} ...")
    retry=0
    warc_zip_path = None
    while warc_zip_path is None and retry < 3:
        if retry > 0:           
            logger.info(f"Retry download chunk {chunk:03}, volume {volume:03} ...")
            time.sleep(60) # 有可能会超过最大链接数量，sleep一下再下载
        warc_zip_path = download_warc(warc_table, chunk, volume, warc_dir, retry>0)
        retry +=1    
    if warc_zip_path:
        return chunk, volume, warc_zip_path
    else:
        logger.error(f"Donwload error, chunk {chunk:03}, volume {volume:03}.")
        return None

def extract_task(chunk, volume, warc_zip_path, out_dir:Path):    
    volume_dir = out_dir / f"src/{chunk:03}/{volume:03}"
    volume_dir.mkdir(parents=True, exist_ok= True)  
    warc_dir = out_dir / "warc" 
    warc_dir.mkdir(exist_ok=True, parents=True)
    tbar = tqdm() 
    logger.info(f"Unziping chunk {chunk:03}, volume {volume:03} ...")
    warc_path = unzip_warc(chunk, volume, warc_zip_path, warc_dir)
    # os.remove(warc_zip_path)
    tbar.set_description(f"extracting chunk {chunk:03}, volume {volume:03}")    
    extract_html(warc_path, volume_dir, tbar)        
    os.remove(warc_path)
    return volume_dir

SCREENSHOT_EVENT =  multiprocessing.Event()  
def format_task(chunk, volume, warc_zip_path, save_dir, html_range, css_range, screenshot_queue): 
    volume_dir = extract_task(chunk, volume, warc_zip_path, Path(save_dir))  
    html_paths = glob.glob(str(volume_dir / '*.html'))  
    tbar = tqdm(range(len(html_paths)))
    logger.info(f"Formating chunk {chunk:03}, volume {volume:03}.")
    tbar.set_description_str(f'formating chunk {chunk:03}, volume {volume:03}:') 
    formated_dir = Path(save_dir) / "formated"    
    def max_id(dir_path, chunk, volume):
        t = dir_path / f"{chunk:03}/{volume:03}_*"
        items = glob.glob(str(t))
        if len(items) == 0:
            return -1
        items = [int(i.split('_')[-1]) for i in items]
        return max(items) 
    try:
        max_id_processed = max_id(formated_dir, chunk, volume)
    except:
        max_id_processed = -1
    for idx,path in enumerate(html_paths):
        tbar.update(1)
        try:
            # 保存目录已经存在则跳过
            if max_id_processed > idx:
                continue            
            html = format(path,html_range,css_range)
            if html is not None:
                screenshot_queue.put((chunk, volume, idx, html))
                SCREENSHOT_EVENT.set()
        except Exception as e:
            exception_info = traceback.format_exc()
            if 'TIPS' not in exception_info:
                logger.debug(exception_info)
            continue      
    shutil.rmtree(volume_dir)
    logger.info(f"Formating done chunk {chunk:03}, volume {volume:03}.")
        
def start_screenshot_procs(port, proc_num, save_dir,  ratio_range, proxy):
    procs = []
    for _ in range(proc_num):
        cmd = f"python scripts/data_cc_pipeline/screenshot_server.py -pt {port} -o {save_dir} -r {ratio_range} -p {proxy}"
        p = subprocess.Popen(cmd, shell=True)
        procs.append(p)
    return procs

def send_cmd(sd:socket.socket, cmd, obj):
    def wait_ok():
        ret = sd.recv(1024).decode('utf-8')
        if ret != "ack":
            raise ValueError("Protoco error.")
    sd.send(f"{cmd}".encode('utf-8'))
    wait_ok()   
    msg = json.dumps(obj).encode('utf-8')
    sd.send(f"{len(msg)}".encode('utf-8'))
    wait_ok()
    sd.send(msg)
    wait_ok()   
             
def connection_worker(client:socket.socket, connection_id, screenshot_queue):
    SCREENSHOT_EVENT.wait()
    while(SCREENSHOT_EVENT.is_set() or not screenshot_queue.empty()):
        try:
            chunk, volume, idx, html = screenshot_queue.get(timeout = 5)   
            #logger.info(f"{connection_id} get {chunk} {volume} {idx} len {len(html)}")            
            obj={"chunk":chunk, "volume":volume, "idx":idx, "html":html}
            send_cmd(client, "screenshot", obj)
            # 等待处理结果
            res = client.recv(256).decode('utf-8')
            screenshot_queue.task_done()
        except queue.Empty:
            time.sleep(0.5)    
        except:
            screenshot_queue.task_done()
            exception_info = traceback.format_exc()
            if 'TIPS' not in exception_info:
                logger.debug(exception_info)
            continue
    
    send_cmd(client, "exit", {"content":""})    
    client.close()
    logger.info(f"Client socket {connection_id} closed.")

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('',0)) #绑定到一个空地址和一个随机端口
    addr, port = s.getsockname()
    s.close() #关闭socket
    return port

def start_connections(port, num_clients, screenshot_queue):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(num_clients)
    logger.info(f"Listening on port {port}")

    connection_id = 0
    while connection_id < num_clients:
        client, addr = server.accept()
        logger.info(f"Accepted connection from: {addr[0]}:{addr[1]}, start thread {connection_id}.")
        client_handler = Thread(target=connection_worker, args=(client, connection_id, screenshot_queue), name=f"data_client_{connection_id}")
        client_handler.start()
        connection_id += 1
    logger.info("Close the server client.")
    server.close()   

def main(chunk, volume_range, save_dir, html_range, css_range, ratio_range, screenshot_workers, cpu_usage, proxy):
    downloader = MulThreading(5, logger.error)
    formator = MultiProcessor("Formator", max(1, int(multiprocessing.cpu_count()*cpu_usage)), logger.error)
    screenshot_queue = multiprocessing.Manager().Queue(screenshot_workers*3)
    num_clients = screenshot_workers
    port = get_free_port()
    screenshot_procs = start_screenshot_procs(port, num_clients, str(save_dir), ratio_range, proxy)
    start_connections(port, num_clients, screenshot_queue)
    
    def quit(force=False):    
        downloader.shutdown(force)
        logger.info("Downloader shutdown.")
        formator.shutdown(force)        
        logger.info(f"Formator shutdown. queue size :{screenshot_queue.qsize()}, empty {screenshot_queue.empty()}.")
        try:
            logger.info(f"unfinished size {screenshot_queue.unfinished_tasks}")
        except:
            logger.error(f"{traceback.format_exc()}")
            
        if force:
            while not screenshot_queue.empty():
                screenshot_queue.get()
        else:
            screenshot_queue.join()
        logger.info("screenshot_queue clear.")
        SCREENSHOT_EVENT.clear()
        for p in screenshot_procs:
            p.wait() 
        logger.info("screenshot procs closed.")
            
    def signal_handler(signal, frame):
        logger.info(f'signal {signal} recieved, exit.')         
        for p in multiprocessing.active_children():            
            # 获取堆栈信息并写入文件
            # 杀死子进程
            os.kill(p.pid, signal.SIGKILL)
            # 退出主进程
        os._exit(1)       

    # 设置信号处理程序
    signal.signal(signal.SIGINT, signal_handler) 
   
    def download_done(res):
        if res is None:   
            return
        chunk, volume, warc_zip_path = res 
        logger.info(f"Download done chunk {chunk:03}, volume {volume:03}.")
        formator.add_task(format_task, (chunk, volume, warc_zip_path, save_dir, html_range, css_range, screenshot_queue), None)      
    
    for volume in volume_range: 
        downloader.add_task(download_task, (chunk, volume, Path(save_dir)), download_done)
        
    quit(False)    

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", "-ch", type=int, default="0") # 0-100
    parser.add_argument("--volume_range", "-v", type=str, default="0-1") # 0-900
    parser.add_argument("--save_dir", "-o", type=str, default="./data")
    parser.add_argument("--html_range", "-ht", type=str, default="128-2560")
    parser.add_argument("--css_range", "-c", type=str, default="128-4096")
    parser.add_argument("--ratio_range", "-r", type=str, default="0.5-2")   
    parser.add_argument("--screenshot_workers", "-sn", type=int, default=1) 
    parser.add_argument("--cpu_usage", "-cu", type=float, default=0.95) 
    parser.add_argument("--proxy", "-p", type=str, default="http://127.0.0.1:7890")

    args = parser.parse_args()
    logger.info(f"args :{args}")
    volumes_range = [int(x) for x in args.volume_range.split('-')]
    volumes_range = range(volumes_range[0], volumes_range[1])      
    html_range = [int(x) for x in args.html_range.split('-')]
    css_range = [int(x) for x in args.css_range.split('-')] 

    main(args.chunk, volumes_range, args.save_dir,  html_range, css_range, args.ratio_range, \
         min(multiprocessing.cpu_count(), args.screenshot_workers), args.cpu_usage, args.proxy)