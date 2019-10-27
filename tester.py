import torch
from constants import DEVICE


def test_net(net, loader):
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    count = 0

    net.to(DEVICE)
    with torch.no_grad():
        for data in loader:
            count += 1
            if count % 200 == 0:
                print(count)
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            _, predicted_top1 = torch.max(outputs.data, 1)
            _, predicted_top5 = outputs.topk(k=5, dim=1)
            total += labels.size(0)
            correct_top1 += (predicted_top1 == labels).sum().item()

            for target, label in zip(predicted_top5.tolist(), labels.tolist()):
                if label in target:
                    correct_top5 += 1

    return {'TOP1': 100 * correct_top1 / total,
            'TOP5': 100 * correct_top5 / total}
    # print('Accuracy of the network: TOP 1: {}, TOP 5: {} %'.format(
    #     100 * correct_top1 / total, 100 * correct_top5 / total))
