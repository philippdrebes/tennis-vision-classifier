import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image

from src.models.auto_encoder import ConvAutoencoder
from src.models.cnn import TennisCNN

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

model_type = "cnn"  # Replace with "cnn" or "autoencoder"


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

criterion = torch.nn.MSELoss()
target_class_index = 0


# arch = Architecture(model)


def classify_image(image_path):
    # Load the image
    frame = np.asarray(Image.open(image_path))

    with torch.no_grad():
        # Preprocess the frame
        frame_processed = transform(frame)
        frame_processed = frame_processed.unsqueeze(0)  # Add batch dimension
        frame_processed = frame_processed.to(device)

        # Make a prediction
        outputs = model(frame_processed)

        def predict_cnn(outputs):
            _, predicted = torch.max(outputs, 1)
            return predicted.item() == target_class_index

        def predict_ae(outputs):
            upper_threshold = 0.6
            loss = criterion(frame_processed, outputs)
            return True if loss <= upper_threshold else False

        def predict(outputs):
            if model_type == "autoencoder":
                return predict_ae(outputs)
            else:
                return predict_cnn(outputs)

        return int(predict(outputs))


data = pd.read_csv('../data/index_image_filepaths.csv')
data[f'classifier_class_{model_type}'] = data['image_relative_filepath'].apply(classify_image)

data.to_csv('../data/index_image_filepaths.csv', index=False)
