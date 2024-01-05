from PIL import ExifTags
from datetime import datetime
from torchvision import transforms
import os
import json
import torch

def get_weekly_log_filename():
    return datetime.now().strftime("%Y_week_%U_log.txt")

def save_log(id, status):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_message = f"{timestamp}: ID={id}, Status={status}\n"
    weekly_log_file = get_weekly_log_filename()

    with open(f"./logs/{weekly_log_file}", "a") as log_file:
        log_file.write(log_message)

def save_image(id, img_tensor):
    if img_tensor is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        to_pil = transforms.ToPILImage()
        img_pil = to_pil(img_tensor)

        face_image_path = f"./cropped_faces/{id}/{id}_{timestamp}.png"
        os.makedirs(os.path.dirname(face_image_path), exist_ok=True)
        img_pil.save(face_image_path)



def save_feature(id, img_aligned):

    img_aligned_list = img_aligned.tolist()

    # 创建 feature 目录（如果不存在）
    if not os.path.exists('features'):
        os.makedirs('features')

    # 创建一个文件名，基于给定的 id
    filename = os.path.join('features', f"{id}.json")

    # 将特征数据保存为 JSON 格式
    with open(filename, 'w') as file:
        json.dump(img_aligned_list, file)

    print(f"Feature saved for ID: {id}")


def correct_image_orientation(img):
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    exif = img._getexif()

    if exif is not None:
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)

    return img
