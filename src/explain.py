import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import shap
from PIL import Image

from src.models.auto_encoder import ConvAutoencoder
from src.models.cnn import TennisCNN

transform1 = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
])
transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
])


# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
#
#
def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


#
#
# transform = [
#     transforms.Lambda(nhwc_to_nchw),
#     transforms.Lambda(lambda x: x * (1 / 255)),
#     transforms.Normalize(mean=mean, std=std),
#     transforms.Lambda(nchw_to_nhwc),
# ]
#
# inv_transform = [
#     transforms.Lambda(nhwc_to_nchw),
#     transforms.Normalize(
#         mean=(-1 * np.array(mean) / np.array(std)).tolist(),
#         std=(1 / np.array(std)).tolist(),
#     ),
#     transforms.Lambda(nchw_to_nhwc),
# ]
#
# transform = transforms.Compose(transform)
# inv_transform = transforms.Compose(inv_transform)

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using {device} device")
device = "cpu"

model_type = "cnn"  # Replace with "cnn" or "autoencoder"

target_class_index = 0
criterion = torch.nn.MSELoss()


def load_cnn(device=device, model_path='model.pth'):
    model = TennisCNN(512, 256).to(device)
    model_state, optimizer_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    print("Model loaded.")
    return model


def load_ae(device=device, model_path='autoencoder.pth'):
    model = ConvAutoencoder().to(device)
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    print("Model loaded.")
    return model


if model_type == "autoencoder":
    model = load_ae()
else:
    model = load_cnn()

image = np.asarray(
    Image.open(
        '../video/frames/test/play/Emma Raducanu vs Leylah Fernandez Full Match ｜ 2021 US Open Final [F99Kz2eptqM]_frame_0288.jpg'
    )
)

image_processed = transform1(image)

# image_processed = image_processed.unsqueeze(0)  # Add batch dimension
image_processed = image_processed.to(device)


def predict(img: np.ndarray) -> torch.Tensor:
    img = nhwc_to_nchw(torch.Tensor(img))
    img = transform(img)
    # img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)

    # _, predicted = torch.argmax(output, 1)
    return output


dataset = datasets.ImageFolder(root=os.path.join('../video/frames/test'), transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

class_names = ['play', 'waiting']

# batch = next(iter(loader))
# images, _ = batch
# images.size()
#
# background = images[:4]
# e = shap.DeepExplainer(model, background)
#
# n_test_images = 4
# test_images = images[:n_test_images]
# shap_values = e.shap_values(images)
#
# # rehspae the shap value array and test image array for visualization
# shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
# test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
#
# # plot the feature attributions
# shap.image_plot(shap_numpy, -test_numpy)

# Explain
topk = 2
batch_size = 50
n_evals = 10000

# define a masker that is used to mask out partitions of the input image.
masker_blur = shap.maskers.Image("blur(128,128)", image_processed.shape)

# create an explainer with model and image masker
explainer = shap.Explainer(predict, masker_blur, output_names=class_names)

# feed only one image
# here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
shap_values = explainer(
    image_processed.unsqueeze(0),
    max_evals=n_evals,
    batch_size=batch_size,
    outputs=shap.Explanation.argsort.flip[:topk],
)

# rehspae the shap value array and test image array for visualization
shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(image_processed.numpy(), 1, -1), 1, 2)

# plot the feature attributions
shap.image_plot(shap_numpy, -test_numpy)

# shap_values.data = inv_transform(shap_values.data).cpu().numpy()
# shap_values.values = [val for val in np.moveaxis(shap_values.values, -1, 0)]
#
# shap.image_plot(
#     shap_values=shap_values.values,
#     pixel_values=shap_values.data,
#     labels=shap_values.output_names,
#     true_labels=[class_names[0]],
# )
