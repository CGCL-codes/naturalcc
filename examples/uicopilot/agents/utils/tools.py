from PIL import Image


def crop_image(image, bbox):
    # 获取图片的宽度和高度
    width, height = image.size

    # 计算截取区域的坐标
    left = bbox[0] * width
    top = bbox[1] * height
    right = bbox[2] * width
    bottom = bbox[3] * height

    # 定义截取区域
    crop_area = (left, top, right, bottom)

    # 使用crop方法截取图片
    cropped_image = image.crop(crop_area)

    return cropped_image

def assemble_node_list(node_list):
    prompt = ''
    for node in node_list:
        prompt += f"""
## {node['name']}
### bbox
({', '.join([str(x) for x in node['bbox']])})
### html
{node['html']}
"""
    return prompt