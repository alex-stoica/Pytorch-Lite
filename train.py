from constants import EPOCHS, LR, TBOARD_PATH, LOAD_MODEL, SIZE, DEVICE
import torch
import torch.optim as optim
from torch import nn
from loaders.dataset import get_classification_dataset, get_car_detection_dataset
from torch.utils.tensorboard import SummaryWriter
from tester import test_net
import time
import os
from datetime import datetime
from initializer import load_model


def main(test_train=True, test_test=True):
    trainloader, validationloader, testloader = get_classification_dataset()
    net = load_model('mnas_xs')

    dt_string = datetime.now().strftime("_%d_%m_%Y_%H_%M")
    writer = get_tboard_writer(net, dt_string, TBOARD_PATH)

    params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    print('Training net ({} parameters, device = {})...'.format(params, DEVICE))

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        running_loss = 0.0
        t = time.time()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            # optimizer = scheduler(optimizer, epoch)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += outputs.shape[0] * loss.item()
            running_loss += loss.item()
            if i % 30 == 29:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 30))
                running_loss = 0.0

        print('Testing on validation dataset...')
        metric = test_net(net, validationloader)

        writer.add_scalar('training loss', epoch_loss / i, epoch + 1)
        writer.add_scalar('valid TOP1 acc', metric['TOP1'], epoch + 1)
        writer.add_scalar('valid TOP5 acc', metric['TOP5'], epoch + 1)

        print("Epoch {} finished after {:.1f} seconds. Training loss = {}. Validation loss = "
              "{}(top1), {}(top5)".format(epoch + 1, time.time() - t, loss.item(), metric['TOP1'], metric['TOP5'])
              )
        save_model(net, epoch, dt_string)

    print('Finished Training')

    if test_train:
        print('Testing on training set...', end="")
        print(test_net(net, trainloader))
    if test_test:
        print('Testing on test set...', end="")
        print(test_net(net, testloader))


def get_tboard_writer(net, dt_string, tboard_path: str = TBOARD_PATH):
    summary_writer_path = os.path.join(tboard_path, net.name + dt_string)
    if not os.path.exists(summary_writer_path):
        os.mkdir(summary_writer_path)
    writer = SummaryWriter(os.path.join(summary_writer_path))

    return writer


def save_model(net, epoch, dt_string):
    saving_path = os.path.join("saved_models", net.name + dt_string)
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    name = net.name + str(epoch + 1) + ".pth"
    torch.save(
        net.state_dict(),
        os.path.join(saving_path, name)
    )


if __name__ == '__main__':
    main()
