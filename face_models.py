from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ExifTags
import torch
import io

# Global variable to set the computation device based on CUDA availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def getDevice():
    """
    Returns the global computation device.
    """
    return device

def load_face_models():
    """
    Loads and returns the MTCNN and InceptionResnetV1 models.
    Returns:
        MTCNN: The MTCNN model for face detection.
        InceptionResnetV1: The Inception Resnet V1 model for face embeddings.
    """
    # Initialize MTCNN with specific configurations
    mtcnn = MTCNN(
        image_size=160,
        margin=14,
        device=device,
        selection_method='center_weighted_size'
    )
    
    # Load the InceptionResnetV1 model pre-trained on 'vggface2'
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    return mtcnn, resnet

def get_face_embedding(image_bytes, mtcnn, resnet):
    """
    Processes an image to extract face embeddings.
    Parameters:
        image_bytes (bytes): The image in byte format.
        mtcnn (MTCNN): The MTCNN model for face detection.
        resnet (InceptionResnetV1): The model for generating face embeddings.
    Returns:
        tuple: A tuple containing the status, face embedding, and aligned image.
    """

    # Try to open and process the image
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = correct_image_orientation(img)
    except Exception as e:
        return "image_error", None, None

    # Detect faces and return probabilities
    img_aligned, prob = mtcnn(img, return_prob=True)
    if img_aligned is None:
        return "no_face", None, None
    elif prob < 0.99:
        return "low_quality", None, img_aligned
    else:
        # Compute the face embedding using the ResNet model
        img_embedding = resnet(img_aligned.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        return "success", img_embedding, img_aligned

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