import os
import requests
requests.packages.urllib3.disable_warnings() 

def download_file(url, filename, rnew=False, chunk_size = 1024*1024, proxies={}):    
    if not rnew and os.path.exists(filename):
        # 如果文件存在，则获取本地文件的大小
        local_file_size = os.path.getsize(filename)
    else:
        local_file_size = 0

    # 发送HEAD请求，获取远程文件的大小
    response = requests.head(url)
    remote_file_size = int(response.headers['content-length'])    

    if remote_file_size == local_file_size:
        return True
    headers = {'Range': 'bytes=%d-' % local_file_size, 'Connection': 'close'}
    # 发送GET请求，指定Range，从本地文件大小的位置开始下载，实现断点续传
    response = requests.get(url, headers=headers, stream=True, verify=False, proxies=proxies)

    # 使用with语句打开文件，使用二进制写入模式，这样可以保证即使程序异常退出，文件也会被关闭
    with open(filename, "wb" if rnew else  'ab') as fp:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fp.write(chunk)
           
    return os.path.getsize(filename) == remote_file_size


