import torch
import torch.nn as nn
import torch.optim as optim
import ModelFunctions as mf


def trainModel(model, train_loader):
    loss_function = nn.NLLLoss()  # negative log likelihood loss
    optimizer1 = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    num_epochs = 10
    for epoch in range(num_epochs):
        loss_ = 0
        for temp_images, labels in train_loader:
            # Flatten the input images of [28,28] to [1,784]
            images = []
            for i in range(len(temp_images)):
                images.append(mf.apply_transforms(temp_images[i], size=29).view(29 * 29))

            images = torch.stack(images)
            # Forward Pass
            output = model(images)
            # Loss at each iteration by comparing to target(label)
            loss = loss_function(output, labels)

            # Backpropogating gradient of loss
            optimizer1.zero_grad()
            loss.backward()

            # Updating parameters(weights and bias)
            optimizer1.step()

            loss_ += loss.item()
        print("Epoch{}, Training loss:{}".format(epoch, loss_ / len(train_loader)))


def save_model(model, path):
    torch.save(model.state_dict(), path)


def accuracy_test(model, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for temp_images, labels in test_loader:
            images = []
            for i in range(len(temp_images)):
                images.append(mf.apply_transforms(temp_images[i], size=29).view(29 * 29))
            images = torch.stack(images)
            out = model(images)
            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Testing accuracy: {} %'.format(100 * correct / total))

