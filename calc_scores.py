"""
Calculate the FID and IS for generated images
"""

from scipy.linalg import sqrtm
import numpy as np
import torch
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
import torch.nn.functional as F


def calculate_inception_score(images, inception_model, num_splits=10):
    # Assume images is a PyTorch tensor of shape (N, C, H, W)
    # and normalized as expected by Inception v3

    # Get the probability distribution for labels for each image
    preds = F.softmax(inception_model(images), dim=1)
    preds = preds.detach().cpu().numpy()

    # Compute the marginal distribution
    marginal = np.mean(preds, axis=0)
    
    # Compute the KL divergence and Inception score
    scores = []
    for i in range(num_splits):
        part = preds[(i * preds.shape[0] // num_splits):((i + 1) * preds.shape[0] // num_splits), :]
        kl = part * (np.log(part) - np.log(marginal))
        kl_mean = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl_mean))

    # Return the mean and standard deviation of the IS
    return np.mean(scores), np.std(scores)

if __name__ == "__main__":
    # Load the Inception v3 model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.cuda()
    inception_model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception v3 requires images of size 299x299
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    

    # Load the generated images
    fake_images = np.load("").cuda()
    fake_images = torch.stack([transform(image) for image in fake_images], dim=0)

    is_mean, is_std = calculate_inception_score(fake_images, inception_model)
    print(f"Inception Score: {is_mean} ± {is_std}")

    # Calculate the FID