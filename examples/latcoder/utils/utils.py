import io
import os
import hashlib
import re
import base64


def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = buffered.getvalue()
    image_base64 = base64.b64encode(img_str).decode('utf-8')
    return image_base64

def image2md5(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_data = image_bytes.getvalue()
    md5_hash = hashlib.md5(image_data)
    md5_hex = md5_hash.hexdigest()
    return str(md5_hex)
    
def remove_code_markers(code):
    #  ``` 
    cleaned_code = re.sub(r'^```html\s*|\s*```$', '', code, flags=re.MULTILINE)
    return cleaned_code

def crop_image(image, bbox):
    # 
    width, height = image.size

    # 
    left = bbox[0] * width
    top = bbox[1] * height
    right = bbox[2] * width
    bottom = bbox[3] * height

    # 
    crop_area = (left, top, right, bottom)

    # crop
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
