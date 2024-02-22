import cv2
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np

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

model_type = "autoencoder"  # Replace with "cnn" or "autoencoder"


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

video_path = 'test.mp4'
output_video_path = f'out_{model_type}.mp4'
output_video_with_overlay_path = f'out_{model_type}_overlay.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties to use with the VideoWriter
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 file
fourcc2 = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
out_overlay = cv2.VideoWriter(output_video_with_overlay_path, fourcc2, fps, (frame_width, frame_height))

target_class_index = 0  # Replace X with the index of your target class

pip_size = (178, 100)  # Width, Height in pixels
pip_position = (20, 20)

show_pip = True
show_timecode = True

criterion = torch.nn.MSELoss()


# arch = Architecture(model)


def frame_to_timecode(frame_number, fps):
    total_seconds = int(frame_number / fps)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


frame_count = 0
saved_frame_count = 0
with torch.no_grad(), tqdm(total=total_frames, desc="Processing Video") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame_processed = transform(frame)
        frame_processed = frame_processed.unsqueeze(0)  # Add batch dimension
        frame_processed = frame_processed.to(device)
        frame_count += 1

        # Make a prediction
        outputs = model(frame_processed)


        def predict_cnn(outputs):
            _, predicted = torch.max(outputs, 1)
            return predicted.item() == target_class_index


        def predict_ae(outputs):
            upper_threshold = 0.8
            loss = criterion(frame_processed, outputs)
            return True if loss <= upper_threshold else False


        def predict(outputs):
            if model_type == "autoencoder":
                return predict_ae(outputs)
            else:
                return predict_cnn(outputs)


        if predict(outputs):
            out.write(frame)
            out_frame_with_overlay = frame
            saved_frame_count += 1
        else:
            black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            out_frame_with_overlay = black_frame

        if show_pip:
            # Overlay PiP
            pip_image = cv2.resize(frame, pip_size, interpolation=cv2.INTER_AREA)
            x_offset, y_offset = pip_position
            out_frame_with_overlay[y_offset:y_offset + pip_size[1], x_offset:x_offset + pip_size[0]] = pip_image

        if show_timecode:
            # Calculate and overlay timecode
            timecode = frame_to_timecode(frame_count, fps)
            cv2.putText(out_frame_with_overlay, timecode, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out_overlay.write(out_frame_with_overlay)
        pbar.update(1)

print(
    f"Total frames processed: {total_frames}, Frames saved to video: {saved_frame_count}, Shortened to: {(saved_frame_count / total_frames) * 100:.2f}%")

# Release everything when job is finished
cap.release()
out.release()
out_overlay.release()
cv2.destroyAllWindows()

# arch.save('out.tex')
