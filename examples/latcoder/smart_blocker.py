import os
from tqdm import tqdm
from PIL import Image, ImageDraw
from datasets import load_dataset
from PIL import Image, ImageDraw
from PIL import Image, ImageDraw
import numpy as np
import json
import easyocr
import cv2
import torch

def blocker(image):
    for i in tqdm(range(0,1)):
        # 先进行清空操作
        # bbbox = []
        bbbox.clear()

        root = {
            'image': image,
            'bbox': [0,0,image.width,image.height],
            'ox': 0,
            'oy': 0
        }

        print('Start')
        results, preview = ocr_with_easyocr(image, merge_threshold=20)
        bboxs = list(map(lambda x:x[0], results))
        print('OCR Done')
        treeSplit(root, mceil=5, mblock=50, max_deep=3, skip=10, bboxs=bboxs)
        print('Split Done')

        fnode = True
        max_iter = 100
        iter_count = 0
        min_area = 300*300
        min_edge = 300

        while fnode:
            fnode = tryFind(root)
            if not fnode:
                break
            fnode['tried'] = True
            flag = True
            while flag: 
                for idx, node in enumerate(fnode['children']):
                    if idx >= len(fnode['children'])-1:
                        flag = False
                        break
                    next_node = fnode['children'][idx+1]
                    bbox = node['bbox']
                    next_bbox = next_node['bbox']
                    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                    next_area = (next_bbox[2]-next_bbox[0])*(next_bbox[3]-next_bbox[1])
                    if node.get('tried', False) or next_node.get('tried', False):
                        continue
                    if (area < min_area or next_area < min_area) or (node['pure'] and not next_node['children']) or (next_node['pure'] and not node['children']):
                        mergeBros(fnode, idx, idx+1)
                        break
                    if bbox[2]-bbox[0] == next_bbox[2]-next_bbox[0]:
                        if bbox[3]-bbox[1]<min_edge and next_bbox[3]-next_bbox[1]<min_edge:
                            mergeBros(fnode, idx, idx+1)
                            break
                    if bbox[3]-bbox[1] == next_bbox[3]-next_bbox[1]:
                        if bbox[2]-bbox[0]<min_edge and next_bbox[2]-next_bbox[0]<min_edge:
                            mergeBros(fnode, idx, idx+1)
                            break

            if len(fnode['children']) == 1:
                fnode['children'] = []
                fnode['tried'] = False

            iter_count += 1
            if iter_count >= max_iter:
                break
        
        image2 = image.copy()
        drawSplitTree(image2, root)
        print(bbbox)

        bboxes = remove_contained_bboxes(bbbox)
        if len(bboxes) == 0:
            bboxes = [[0,0,image.width,image.height]]

        # bboxes = length2propotion(image,bbbox[:])
        print(type(image))
        # image 是类型为<class 'PIL.PngImagePlugin.PngImageFile'>
        # bbbox = [[1, 1, 1279, 78], [1, 80, 1279, 133], [1, 135, 1279, 203], [1, 205, 1279, 258], [1, 260, 1279, 648], [1, 650, 1279, 713], [1, 715, 1279, 764], [1, 766, 1279, 834], [1, 836, 1279, 1029], [1, 1031, 1279, 1084], [1, 1086, 1279, 1139], [1, 1141, 1279, 1194], [1, 1196, 1279, 1249], [1, 1251, 1279, 1319], [1, 1321, 1279, 1384], [1, 1386, 1279, 1439], [1, 1441, 1279, 1531]]

        crop_images = crop_image_by_bboxes(image,bboxes)
        # 下面还要经过检查去除空白
        bboxes = save_cropped_images(crop_images=crop_images,save_path = "/mnt/silver/uiagent/function_agent_gemini",bboxes=bboxes)
        print(len(bbbox))
        print(len(bboxes))
        print("这是bboxes")
        print(bboxes)

        final_crop_image = drawwhole(image=image,bboxes=bboxes)
        return length2propotion(image2=image,bboxes=bboxes) , final_crop_image
    

bbbox = []

def drawOnImage(image, bbox, ox=0, oy=0, copy=False, padding=2):
    # if copy:
    #     image = image.copy()
    # draw = ImageDraw.Draw(image)
    # draw.line([(bbox[0]+ox+padding, bbox[1]+oy+padding), (bbox[0]+ox+padding, bbox[3]+oy-padding)], fill="red", width=2)
    # draw.line([(bbox[2]+ox-padding, bbox[1]+oy+padding), (bbox[2]+ox-padding, bbox[3]+oy-padding)], fill="red", width=2)
    # draw.line([(bbox[0]+ox+padding, bbox[1]+oy+padding), (bbox[2]+ox-padding, bbox[1]+oy+padding)], fill="red", width=2)
    # draw.line([(bbox[0]+ox+padding, bbox[3]+oy-padding), (bbox[2]+ox-padding, bbox[3]+oy-padding)], fill="red", width=2)
    # return image
    if copy:
        image = image.copy()
    draw = ImageDraw.Draw(image)
    
    # 计算带有偏移和填充的矩形坐标
    padded_bbox = [
        bbox[0] + ox + padding,
        bbox[1] + oy + padding,
        bbox[2] + ox - padding,
        bbox[3] + oy - padding
    ]
    bbbox.append(padded_bbox)
    
    # 直接使用 rectangle 绘制矩形
    draw.rectangle(padded_bbox, outline="red", width=2)
    return image
        
def splitImage(image, mceil=10):
    width, height = image.size
    
    dw = round(width/mceil)
    dh = round(height/mceil)

    x_step = width / dw
    y_step = height / dh

    points = []
    last_color = None
    all_same = True

    for j in range(2, dh):
        row = []
        for i in range(2, dw):
            x = int(i * x_step) - 1
            y = int(j * y_step) - 1
            color = image.getpixel((x, y))
            if all_same and last_color and last_color != color:
                all_same = False
            last_color = color
            row.append({
                "x": x,
                "y": y,
                'color': color,
            })

        points.append(row)

    return points, all_same

        
def breakImage(image, points, axis='x', mblock=50, skip=5, bboxs=[], ox=0, oy=0):
    xs = len(points[0])  # 水平方向采样点数量
    ys = len(points)     # 垂直方向采样点数量
    brs = []             # 存储分割线的列表

    if axis == 'x':  # 按 x 轴方向分割
        for i in range(xs):
            flag = True
            pre_color = None
            for j in range(ys):
                if j<skip or j>ys-skip-1:
                    continue
                p = points[j][i]
                color = p['color']
                x = p['x'] + ox
                y = p['y'] + oy
                over_text = bool(len(list(filter(lambda b: x>=b[0][0] and x<=b[2][0], bboxs))))
                if over_text:
                    flag = False
                    break
                if not pre_color:
                    pre_color = color
                else:
                    if color != pre_color:
                        flag = False
                        break

            if flag:
                x = points[0][i]['x']
                if image.width - x > mblock and (
                    (len(brs) and x - brs[-1][0]['x'] > mblock) or 
                    (not len(brs) and x > mblock)
                ):
                    brs.append([{
                        'x': x,
                        'y': 0
                    }, {
                        'x': x,
                        'y': image.height
                    }])

        # 添加最后一条分割线
        if len(brs):
            brs.append([{
                'x': image.width,
                'y': 0
            }, {
                'x': image.width,
                'y': image.height
            }])

    else:  # 按 y 轴方向分割
        for i in range(ys):
            flag = True
            pre_color = None
            for j in range(xs):
                if j<skip or j>xs-skip-1:
                    continue
                p = points[i][j]
                color = p['color']
                x = p['x'] + ox
                y = p['y'] + oy
                over_text = bool(len(list(filter(lambda b: y>=b[0][1] and y<=b[2][1], bboxs))))
                if over_text:
                    flag = False
                    break
                if not pre_color:
                    pre_color = color
                else:
                    if color != pre_color:
                        flag = False
                        break

            if flag: 
                y = points[i][0]['y']
                if image.height - y > mblock and (
                    (len(brs) and y - brs[-1][1]['y'] > mblock) or 
                    (not len(brs) and y > mblock)
                ):
                    brs.append([{
                        'x': 0,
                        'y': y
                    }, {
                        'x': image.width,
                        'y': y
                    }])

        # 添加最后一条分割线
        if len(brs):
            brs.append([{
                'x': 0,
                'y': image.height
            }, {
                'x': image.width,
                'y': image.height
            }])

    return brs

def applyBreak(image, brs, ox=0, oy=0):
    last_x = 0
    last_y = 0
    cs = []
    for br in brs:
        bbox = (last_x, last_y, br[1]['x'], br[1]['y'])
        c = image.crop(bbox)
        last_x = br[0]['x']
        last_y = br[0]['y']
        cs.append({
            'image': c,
            'bbox': bbox,
            'ox': ox+bbox[0],
            'oy': oy+bbox[1]
        })
    return cs

def mergeCnodes(node, begin, end):
    node['children'][end]['bbox'] = [node['children'][begin]['bbox'][0], node['children'][begin]['bbox'][1], node['children'][end]['bbox'][2], node ['children'][end]['bbox'][3]]
    node['children'] = node['children'][:begin] + node['children'][end:]
    if len(node['children']) == 1:
        node['children'] = []

def treeSplit(node, mceil=10, mblock=50, deep=1, max_deep=10, skip=5, bboxs=[]):
    if deep > max_deep:
        node['children'] = []
        return
    image = node['image']
    ps, all_same = splitImage(image, mceil)
    if all_same:
        node['children'] = []
        return
    brs = breakImage(image, ps, axis='y', mblock=mblock, skip=skip, bboxs=bboxs, ox=node['ox'], oy=node['oy'])
    if not len(brs):
        brs = breakImage(image, ps, axis='x', mblock=mblock, skip=skip, bboxs=bboxs, ox=node['ox'], oy=node['oy'])
    if len(brs):
        cs = applyBreak(image, brs, node['ox'], node['oy'])
        node['children'] = cs
        for cnode in node['children']:
            _, pure = splitImage(cnode['image'], mceil)
            cnode['pure'] = pure
            if cnode['image'].width > mceil and cnode['image'].height > mceil:
                treeSplit(cnode, mceil, mblock, deep=deep+1, max_deep=max_deep, skip=skip, bboxs=bboxs)
    else:
        node['children'] = []
    
def drawSplitTree(image, node, ox=0, oy=0, deep=1):
    for cnode in node['children']:
        drawOnImage(image, cnode['bbox'], ox, oy, padding=deep**2)
        drawSplitTree(image, cnode, ox=ox+cnode['bbox'][0], oy=oy+cnode['bbox'][1], deep=deep+1)

def tryFind(node):
    # 检查当前节点是否有子节点
    if 'children' in node and isinstance(node['children'], list) and node['children']:
        # 检查当前节点的所有子节点是否都是叶子节点
        all_children_are_leaves = all(
            'children' not in child or len(child['children'])==0 for child in node['children']
        )
        if all_children_are_leaves and not node.get('tried', False):
            return node

        # 如果当前节点不满足条件，递归检查子节点
        for child in node['children']:
            result = tryFind(child)
            if result:
                return result
    
    # 检索条件放松，把已经tried过的也当叶子节点
    if 'children' in node and isinstance(node['children'], list) and node['children']:
        # 检查当前节点的所有子节点是否都是叶子节点
        all_children_are_leaves = all(
            'children' not in child or len(child['children'])==0 or child.get('tried', False) for child in node['children']
        )
        if all_children_are_leaves and not node.get('tried', False):
            return node

        # 如果当前节点不满足条件，递归检查子节点
        for child in node['children']:
            result = tryFind(child)
            if result:
                return result

    # 如果当前节点没有子节点或不满足条件，返回 None
    return None

def tryMerge(node):
    node['tried'] = True
    flag = True
    while flag:
        last_pure_idx = -1
        flag = False
        for idx, cnode in enumerate(node['children']):
            if cnode['pure']:
                if last_pure_idx == -1:
                    last_pure_idx = idx
                elif idx == len(node['children'])-1:
                    mergeCnodes(node, last_pure_idx, idx)
                    flag = True
                    break
            else:
                if last_pure_idx != -1:
                    if idx-last_pure_idx>1:
                        mergeCnodes(node, last_pure_idx, idx-1)
                        flag = True
                        break
                    else:
                        last_pure_idx = -1

def mergeBros(fnode, idx1, idx2):
    node = fnode['children'][idx1]
    next_node = fnode['children'][idx2]
    node['bbox'] = [node['bbox'][0], node['bbox'][1], next_node['bbox'][2], next_node['bbox'][3]]
    node['pure'] = node['pure'] and next_node['pure']
    fnode['children'] = fnode['children'][:idx2] + fnode['children'][idx2+1:]

def merge_bboxs(results, threshold=50):
    while True:
        flag = True
        for idx, item in enumerate(results):
            bbox, text, _ = item
            left, top, right, bottom = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]
            for idx2, item2 in enumerate(results):
                if idx2 == idx:
                    continue
                bbox2, text2, _ = item2
                left2, top2, right2, bottom2 = bbox2[0][0], bbox2[0][1], bbox2[2][0], bbox2[2][1]
                if not (left2 > right + threshold or right2 < left - threshold or bottom2 < top - threshold or top2 > bottom + threshold):
                    left3, top3, right3, bottom3 = min(left, left2), min(top, top2), max(right, right2), max(bottom, bottom2)
                    results[idx] = ([[left3, top3], [right3, top3], [right3, bottom3], [left3, bottom3]], f'{text}\n{text2}', None)
                    results = results[:idx2]+results[idx2+1:]
                    flag = False
                    break
            if not flag:
                break
        if flag:
            return results

def ocr_with_easyocr(pil_image, lang_list=['en', 'ch_sim'], merge_threshold=50):
    reader = easyocr.Reader(lang_list, gpu=False)  # 设置 GPU=False 以使用 CPU

    # 将 PIL.Image 转换为 numpy 数组
    image_np = np.array(pil_image)

    # 使用 easyocr 识别文本
    results = reader.readtext(image_np)

    # 迭代合并临近的边界框
    results = merge_bboxs(results, threshold=merge_threshold)

    # 绘制合并后的边界框
    for item in results:
        bbox, text, _ = item
        top_left = tuple(map(int, bbox[0]))
        top_right = tuple(map(int, bbox[1]))
        bottom_right = tuple(map(int, bbox[2]))
        bottom_left = tuple(map(int, bbox[3]))

        # 绘制边界框
        cv2.polylines(image_np, [np.array([top_left, top_right, bottom_right, bottom_left])], isClosed=True, color=(0, 255, 0), thickness=2)

    # 将 numpy 数组转换回 PIL.Image
    result_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    # 返回识别的文本信息和带 bbox 的图片
    return results, result_image



def length2propotion(image2,bboxes):
    for i in range(len(bboxes)):
        bboxes[i][0] = round(bboxes[i][0] / image2.width, 3)
        bboxes[i][1] = round(bboxes[i][1] / image2.height, 3)
        bboxes[i][2] = round(bboxes[i][2] / image2.width, 3)
        bboxes[i][3] = round(bboxes[i][3] / image2.height, 3)
    return bboxes 

def propotion2length(image,bboxes):
    bbboxes = []
    for bbox in bboxes:
        x1 = int(bbox[0] * image.width)
        y1 = int(bbox[1] * image.height)
        x2 = int(bbox[2] * image.width)
        y2 = int(bbox[3] * image.height)
        bbboxes.append([x1,y1,x2,y2])
    return bbboxes

def crop_image_by_bboxes(image, bboxes):
    """
    根据给定的 bbox 坐标列表裁剪图片。
    
    参数:
    - image: PIL 图像对象 (如 <class 'PIL.PngImagePlugin.PngImageFile'>)
    - bboxes: 坐标列表，每个元素为 [x1, y1, x2, y2]，表示裁剪框的左上角和右下角
    
    返回:
    - cropped_images: 裁剪后的图像列表
    """
    cropped_images = []
    
    for bbox in bboxes:
        # 将坐标分解为 x1, y1, x2, y2
        x1, y1, x2, y2 = bbox
        
        # 裁剪图片并保存到 cropped_images 列表
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_images.append(cropped_image)
    
    return cropped_images

def is_blank_image(image):
    """
    检查图片是否为空白（全白或完全透明）。
    
    参数:
    - image: PIL 图像对象
    
    返回:
    - True 如果是空白图片，False 否则
    """
    # 将图像转换为RGB模式（如果是RGBA模式，去掉透明度）
    image = image.convert("RGB")
    
    # 获取图片的所有像素
    pixels = image.getdata()
    
    # 检查是否所有像素都是白色 (255, 255, 255)
    for pixel in pixels:
        if pixel != (255, 255, 255):  # 白色是 (255, 255, 255)
            return False
    
    return True

def save_cropped_images(crop_images, save_path,bboxes):
    """
    保存裁剪后的图片，如果图片不为空白则保存。
    
    参数:
    - crop_images: 裁剪后的图片列表
    - save_path: 图片保存的路径
    """
    new_bboxes = []
    for index, image in enumerate(crop_images):
        if not is_blank_image(image):  # 如果图片不为空白
            print(f"{index}is not empty")
            # image.save(f"{save_path}/{index}.png")
            # print(f"{save_path}/{index}.png")
            new_bboxes.append(bboxes[index])
        else:
            print(f"Image {index} is blank, not saving.")
    return new_bboxes



def is_contained(bbox1, bbox2):
    """
    判断 bbox1 是否包含 bbox2。
    
    参数:
    - bbox1: 外部矩形 [x1, y1, x2, y2]
    - bbox2: 内部矩形 [x1', y1', x2', y2']
    
    返回:
    - True 如果 bbox1 包含 bbox2，False 否则
    """
    x1, y1, x2, y2 = bbox1
    x1_prime, y1_prime, x2_prime, y2_prime = bbox2
    
    # 判断是否 bbox1 包含 bbox2
    return x1 <= x1_prime and y1 <= y1_prime and x2 >= x2_prime and y2 >= y2_prime

def remove_contained_bboxes(bboxes):
    """
    移除包含的模块，只保留外部模块。
    
    参数:
    - bboxes: 包含多个矩形框的列表 [[x1, y1, x2, y2], ...]
    
    返回:
    - 筛选后的矩形框列表
    """
    result = []
    
    for i, bbox1 in enumerate(bboxes):
        is_contained_flag = False
        for j, bbox2 in enumerate(bboxes):
            if i != j and is_contained(bbox1, bbox2):
                is_contained_flag = True
                break
        
        if not is_contained_flag:
            result.append(bbox1)
    
    return result

def drawwhole(image,bboxes):
    '''
        画出整个分割的样式
    '''
    image2 = image.copy();
    draw = ImageDraw.Draw(image2)
    print(bboxes)
    for bbox in bboxes:
        x1,y1,x2,y2 = bbox
        draw.rectangle([(x1,y1),(x2,y2)],outline="red",width=2)
    return image2