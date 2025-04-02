import io
import os
import hashlib

def image2md5(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_data = image_bytes.getvalue()
    md5_hash = hashlib.md5(image_data)
    md5_hex = md5_hash.hexdigest()
    return str(md5_hex)

def save_result(result_path, image, answer, prediction, duration, imgs=[]):
    md5 = image2md5(image)
    os.makedirs(os.path.join(result_path,f'{md5}'),exist_ok=True)
    image.save(os.path.join(result_path,f'{md5}/image.png'))
    for idx,img in enumerate(imgs):
        img.save(os.path.join(result_path,f'{md5}/{idx}.png'))
    with open(os.path.join(result_path,f'{md5}/answer.html'),'w') as f:
        f.write(answer)
    with open(os.path.join(result_path,f'{md5}/prediction.html'),'w') as f:
        f.write(prediction)
    with open(os.path.join(result_path,f'{md5}/time.csv'),'a+') as f:
        f.write(f'{duration}\n')