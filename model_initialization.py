# Import the InceptionResnetV1 class from the facenet_pytorch module
from facenet_pytorch import InceptionResnetV1

# Initialize the InceptionResnetV1 model with weights pre-trained on the 'vggface2' dataset
resnet = InceptionResnetV1(pretrained='vggface2').eval()
