import torch
import torch.nn as nn
import torch.optim as optim
import ModelFunctions as mf

def train_conv_model(model, train_loader):
    num_epochs = 5
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))


def test_conv_model(model, test_loader):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    # Save the model and plot
    torch.save(model.state_dict(), 'conv_net_model.pt')

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

