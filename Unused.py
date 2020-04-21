# """## 3) Test code that works (demo)
#
# #### Netwerk managers initialiseren en netwerken trainen of inladen
#
# Train the models model1 and model2
# """
#
# #lossFunction = nn.NLLLoss() #negative log likelihood loss
# #optimizer1 = optim.SGD(model1.parameters(), lr=0.003, momentum=0.9)
# #optimizer2 = optim.SGD(model2.parameters(), lr=0.003, momentum=0.9)
# #num_epochs = 10
# #for epoch in range(num_epochs):
# #    loss_ = 0
# #    for images, labels in train_loader:
# #        # Flatten the input images of [28,28] to [1,784]
# #        images = images.reshape(-1, 28*28)
# #
# #        # Forward Pass
# #        output = model1(images)
# #        # Loss at each iteration by comparing to target(label)
# #        loss = lossFunction(output, labels)
# #
# #        # Backpropogating gradient of loss
# #        optimizer1.zero_grad()
# #        loss.backward()
# #
# #        # Updating parameters(weights and bias)
# #        optimizer1.step()
# #
# #        loss_ += loss.item()
# #    print("Epoch{}, Training loss:{}".format(epoch, loss_ / len(train_loader)))
# #    num_epochs = 10
# #for epoch in range(num_epochs):
# #    loss_ = 0
# #    for images, labels in train_loader:
# #        # Flatten the input images of [28,28] to [1,784]
# #        images = images.reshape(-1, 784)
# #
# #        # Forward Pass
# #        output = model2(images)
# #        # Loss at each iteration by comparing to target(label)
# #        loss = lossFunction(output, labels)
# #
# #        # Backpropogating gradient of loss
# #        optimizer2.zero_grad()
# #        loss.backward()
# #
# #        # Updating parameters(weights and bias)
# #        optimizer2.step()
# #
# #        loss_ += loss.item()
# #    print("Epoch{}, Training loss:{}".format(epoch, loss_ / len(train_loader)))
#
# """Save model 1 and 2"""
#
# # torch.save(model1.state_dict(), "mnist_model.pt")
# # torch.save(model2.state_dict(), "mnist_model_seq.pt")
#
# """Accuracy test"""
#
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 784)
#         out = model1(images)
#         _, predicted = torch.max(out, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print('Testing accuracy: {} %'.format(100 * correct / total))
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 784)
#         out = model2(images)
#         _, predicted = torch.max(out, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print('Testing accuracy: {} %'.format(100 * correct / total))
#