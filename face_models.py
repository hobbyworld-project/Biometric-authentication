from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Global variable for device

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def getDevice():
    return device

def load_face_models():
    mtcnn = MTCNN(
    image_size=160,
    margin=14,
    device=device,
    selection_method='center_weighted_size'
    )
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    return mtcnn, resnet

def get_face_embedding(image_bytes, mtcnn, resnet):
    from PIL import Image
    import io
    from utils import correct_image_orientation

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = correct_image_orientation(img)
    except Exception as e:
        return "image_error", None, None

    img_aligned, prob = mtcnn(img, return_prob=True)
    if img_aligned is None:
        return "no_face", None, None
    elif prob < 0.99:
        return "low_quality", None, img_aligned
    else:
        img_embedding = resnet(img_aligned.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        return "success", img_embedding, img_aligned
