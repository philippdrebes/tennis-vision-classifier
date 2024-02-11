from pytorch2tikz import Architecture

import torch
from torchvision import transforms
import imageio.v3 as iio

from src.cnn import TennisCNN

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print('Load model')
model = TennisCNN(512, 256).to(device)
model_state, optimizer_state = torch.load('model.pth', map_location=device)
model.load_state_dict(model_state)
model = model.to(device)
model.eval()
print("Model loaded.")

print('Load data')
img = iio.imread(
    '../video/frames/Roger Federer v Rafael Nadal Full Match ｜ Australian Open 2017 Final [KTCDxjJvs2U]/Roger Federer v Rafael Nadal Full Match ｜ Australian Open 2017 Final [KTCDxjJvs2U]_frame_0170.jpg')

# Preprocess the frame
frame_processed = transform(img)
frame_processed = frame_processed.unsqueeze(0)  # Add batch dimension
frame_processed = frame_processed.to(device)

print('Init architecture')
arch = Architecture(model)

print('Run model')
with torch.inference_mode():
    # for image, _ in data_loader:
    output = model(frame_processed)

print('Write result to out.tex')
arch.save('out.tex')
