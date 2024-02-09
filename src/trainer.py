import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import tempfile

from src.cnn import TennisCNN


def load_data(data_dir="../video/frames"):
    # Augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Just normalization for validation and testing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    dataset_test = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_transform)

    return dataset, dataset_test


def train_tennis(config, data_dir=None):
    net = TennisCNN(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using {device} device")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    # Splitting dataset into train and validation
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    valloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")


def test_accuracy(net, device="cpu"):
    data_dir = os.path.abspath("../video/frames")
    trainset, testset = load_data(data_dir)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("../video/frames")
    load_data(data_dir)
    config = {
        "l1": tune.choice([2 ** i for i in range(5, 10)]),
        "l2": tune.choice([2 ** i for i in range(5, 10)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([2, 4, 8, 16, 32]),
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(partial(train_tennis, data_dir=data_dir)),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_trial = results.get_best_result("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.metrics["accuracy"]))

    best_trained_model = TennisCNN(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main(num_samples=20, max_num_epochs=10, gpus_per_trial=0)

# train_losses = []
# val_losses = []
# train_accuracies = []
# val_accuracies = []
#
# num_epochs = 25
#
# for epoch in range(num_epochs):
#     # Training
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     tkt = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", smoothing=0, mininterval=1.0)
#     for i, (images, labels) in enumerate(tkt):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     train_losses.append(running_loss / len(train_loader))
#     train_accuracies.append(100 * correct / total)
#
#     # Validation
#     model.eval()  # Set the model to evaluation mode
#     val_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#
#     tkv = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", smoothing=0, mininterval=1.0)
#     with torch.no_grad():  # Disable gradient calculation
#         for i, (images, labels) in enumerate(tkv):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         val_losses.append(val_loss / len(val_loader))
#         val_accuracies.append(100 * correct / total)
#
#     print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, '
#           f'Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.2f}%')
#
# torch.save(model.state_dict(), "model.pth")
# Call the function to plot metrics
# plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

# Testing
# model.eval()  # Set the model to evaluation mode
# test_loss = 0.0
# correct_predictions = 0
# total_predictions = 0
# all_predictions = []
# all_labels = []
#
# with torch.no_grad():  # Disable gradient calculation
#     for images, labels in tqdm(test_loader, desc="Testing", smoothing=0, mininterval=1.0):
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         test_loss += loss.item()
#
#         _, predicted = torch.max(outputs.data, 1)
#         all_predictions.extend(predicted.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#         total_predictions += labels.size(0)
#         correct_predictions += (predicted == labels).sum().item()
#
# test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
# avg_test_loss = test_loss / len(test_loader)
# accuracy = correct_predictions / total_predictions * 100
# print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%')
#
# # Compute confusion matrix
# class_names = ['playing', 'waiting']
# cm = confusion_matrix(all_labels, all_predictions)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# disp.plot(cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.show()
