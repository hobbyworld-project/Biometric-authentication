from PIL import ExifTags
from datetime import datetime
from torchvision import transforms
from db_operations import save_embedding
import os
import json

def get_weekly_log_filename():
    """
    Generates a filename for logging based on the current week of the year.
    """
    return datetime.now().strftime("%Y_week_%U_log.txt")

def save_log(id, status):
    """
    Saves a log entry for a given ID and status.
    Parameters:
        id (str): The unique identifier for the subject.
        status (str): The status to be logged.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_message = f"{timestamp}: ID={id}, Status={status}\n"
    weekly_log_file = get_weekly_log_filename()

    with open(f"./logs/{weekly_log_file}", "a") as log_file:
        log_file.write(log_message)

def save_image(id, img_tensor):
    """
    Saves an image tensor as a PNG file.
    Parameters:
        id (str): The unique identifier for the subject.
        img_tensor (Tensor): The image tensor to be saved.
    Returns:
        str: "success" or "fail" based on the operation result.
    """
    if img_tensor is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        to_pil = transforms.ToPILImage()
        img_pil = to_pil(img_tensor)

        face_image_path = f"./cropped_faces/{id}/{id}_{timestamp}.png"
        os.makedirs(os.path.dirname(face_image_path), exist_ok=True)
        img_pil.save(face_image_path)

        return "success"
    else:
        return "fail"

def save_feature(id, img_aligned):
    """
    Saves the aligned image feature as a JSON file and records it in the database.
    Parameters:
        id (str): The unique identifier for the subject.
        img_aligned (Tensor): The aligned image tensor.
    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    img_aligned_list = img_aligned.tolist()

    if not os.path.exists('features'):
        os.makedirs('features')

    filename = os.path.join('features', f"{id}.json")
        
    with open(filename, 'w') as file:
        json.dump(img_aligned_list, file)

    return save_embedding(id, json.dumps(img_aligned_list))

def correct_image_orientation(img):
    """
    Corrects the orientation of an image based on its EXIF data.
    Parameters:
        img (PIL.Image): The image to be corrected.
    Returns:
        PIL.Image: The image with corrected orientation.
    """
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
