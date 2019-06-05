import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

model = models.inception_v3(pretrained=True)
model.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((299, 299)), transforms.ToTensor(),
                                 normalize])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_inception_probs(npy_img):
    """
    Given a numpy array (corresponding to an image) returns the class probabilities output by inception
    """
    img_tensor = transforms(npy_img).unsqueeze(0)
    output = model(img_tensor).detach().numpy()[0]
    probs = softmax(output)
    return probs

def get_inception_logits(npy_img):
    img_tensor = transforms(npy_img).unsqueeze(0)
    logits = model(img_tensor).detach().numpy()[0]
    return logits

