import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.auto_encoder import ConvAutoencoder


def load_data_autoencoder(data_dir="../video/frames"):
    # Transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    dataset_test = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_transform)

    return dataset, dataset_test


def train_tennis_autoencoder(config, data_dir=None):
    net = ConvAutoencoder()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using {device} device")
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    (trainset, testset) = load_data_autoencoder(data_dir)

    # Splitting dataset into train and validation
    # train_size = int(0.8 * len(trainset))
    # val_size = len(trainset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

    # Filter for 'playing' class images
    playing_class_index = trainset.class_to_idx['play']
    indices = [i for i, (_, label) in enumerate(trainset.samples) if label == playing_class_index]
    trainset_filtered = Subset(trainset, indices)

    # Create data loaders
    trainloader = DataLoader(trainset_filtered, batch_size=config["batch_size"], shuffle=True)
    # valloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    testloader = DataLoader(testset, batch_size=config["batch_size"], shuffle=False)

    metrics = defaultdict(list)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        net.train()
        epoch_loss = 0.0
        tkt = tqdm(trainloader, desc=f"Training Epoch {epoch + 1}", smoothing=0, mininterval=1.0)
        for i, (images, labels) in enumerate(tkt):
            # get the inputs; data is a list of [inputs, labels]
            inputs = images.to(device)

            x = inputs  # inputs
            y = inputs  # target

            optimizer.zero_grad()  # prepare gradients
            outp = net(x)  # compute output/target
            loss_val = criterion(outp, y)  # a tensor
            epoch_loss += loss_val.item()  # accumulate for display
            loss_val.backward()  # compute gradients
            optimizer.step()  # update weights

            tkt.set_postfix(loss=loss_val.item())

        metrics['train_loss'].append(epoch_loss)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(trainloader)}")

    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title('Loss')
    ax.plot(metrics['train_loss'])
    plt.show()

    # Validation phase
    net.eval()
    loss_dist = []
    with torch.no_grad():
        tkt = tqdm(testloader)
        for i, (images, labels) in enumerate(tkt):
            x = images.to(device)
            y = net(x)

            # Iterate over each image in the batch
            for idx in range(y.size(0)):  # y.size(0) is the batch size
                # Select the idx-th image in batch, keeping dimension for batch size of 1
                y_img = y[idx].unsqueeze(0)
                x_img = x[idx].unsqueeze(0).to(device)

                # Calculate loss for the single image
                loss = criterion(x_img, y_img)
                loss_dist.append(loss.item())

    loss_sc = []
    for i in loss_dist:
        loss_sc.append((i, i))
    plt.scatter(*zip(*loss_sc))
    plt.axvline(0.3, 0.0, 1)
    plt.show()

    lower_threshold = 0.1
    upper_threshold = 1.1
    plt.figure(figsize=(12, 6))
    plt.title('Loss Distribution')
    sns.displot(loss_dist, bins=100, kde=True, color='blue')
    plt.axvline(upper_threshold, 0.0, 10, color='r')
    plt.axvline(lower_threshold, 0.0, 10, color='b')
    plt.show()

    # You'll need actual labels for your validation images to use them here
    true_labels = [label for _, label in testset]

    # Determine predictions based on loss threshold
    predictions = [1 if loss >= upper_threshold else 0 for loss in loss_dist]

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])

    # Display the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


def make_err_list(model, dataloader, device):
    # Compute Reconstruction Errors
    result_lst = []
    n_features = len(dataloader)

    tkt = tqdm(dataloader)
    for i, (images, labels) in enumerate(tkt):
        x = images.to(device)
        with torch.no_grad():
            y = model(x)  # should be same as x
        err = torch.sum((x - y) * (x - y)).item()  # sse all features
        err = err / n_features  # sort of norm'ed SSE
        result_lst.append((i, err))  # idx of data item, err
    return result_lst


def display_digit(ds, idx, save=False):
    # ds is a PyTorch Dataset
    line = ds[idx]  # tensor
    pixels = np.array(line[0:224])  # numpy row of pixels
    label = np.int(line[224] * 9.0)  # denormalize; like '5'
    print("\ndigit = ", str(label), "\n")

    pixels = pixels.reshape((8, 8))
    for i in range(8):
        for j in range(8):
            pxl = pixels[i, j]  # or [i][j] syntax
            pxl = np.int(pxl * 16.0)  # denormalize
            print("%.2X" % pxl, end="")
            print(" ", end="")
        print("")

    plt.imshow(pixels, cmap=plt.get_cmap('gray_r'))
    if save == True:
        plt.savefig(".\\idx_" + str(idx) + "_digit_" + str(label) + ".jpg", bbox_inches='tight')
    plt.show()
    plt.close()


def main():
    data_dir = os.path.abspath("../video/frames")
    # load_data_autoencoder(data_dir)

    config = {
        "lr": 0.003,
        "weight_decay": 0.0001,
        "batch_size": 32,
        "epochs": 10
    }

    train_tennis_autoencoder(config, data_dir)


if __name__ == "__main__":
    main()
