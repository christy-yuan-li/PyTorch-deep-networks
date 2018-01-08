import torch
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm
import torchnet as tnt
import sys
sys.path.append('../')
import models
import transforms
import Dataset

NUM_EPOCHS = 500
BATCH_SIZE = 16
NUM_CLASSES = 2
size = 256
batch_size = 16
type = 'XR_ELBOW'
augment = True

model = models.CapsuleNet()
# model.load_state_dict(torch.load('epochs/epoch_327.pt'))
model.cuda()
print("# parameters:", sum(param.numel() for param in model.parameters()))
optimizer = Adam(model.parameters())

engine = Engine()
meter_loss = tnt.meter.AverageValueMeter()
meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                 'columnnames': list(range(NUM_CLASSES)),
                                                 'rownames': list(range(NUM_CLASSES))})
ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

capsule_loss = models.CapsuleLoss()



def get_iterator(mode):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    transform_augment = transforms.Compose([
        # transforms.RandomResizedCrop(args.size, scale=(0.8, 1.2)),  # random scale 0.8-1 of original image area, crop to args.size
        transforms.RandomResizedCrop(size),
        transforms.RandomRotation(15),  # random rotation -15 to +15 degrees
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform = transforms.Compose([transforms.Resize((size, size)),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    if mode:
        dataset = Dataset.MURA(split="train", transform=(transform_augment if augment else transform), type=type)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             **kwargs)
    else:
        dataset = Dataset.MURA(split="test", transform=transform, type=type)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             **kwargs)
    return loader


def processor(sample):
    data, labels, studyindex, path, training = sample

    labels = torch.LongTensor(labels)
    labels = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

    data = Variable(data).cuda()
    labels = Variable(labels).cuda()

    if training:
        classes, reconstructions = model(data, labels)
    else:
        classes, reconstructions = model(data)

    loss = capsule_loss(data, labels, classes, reconstructions)

    return loss, classes


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()


def on_sample(state):
    state['sample'].append(state['train'])


def on_forward(state):
    meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
    confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
    meter_loss.add(state['loss'].data[0])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

    reset_meters()

    engine.test(processor, get_iterator(False))
    test_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    confusion_logger.log(confusion_meter.value())

    print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])

    # Reconstruction visualization.

    test_sample = next(iter(get_iterator(False)))

    ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)
    _, reconstructions = model(Variable(ground_truth).cuda())
    reconstruction = reconstructions.cpu().view_as(ground_truth).data

    ground_truth_logger.log(
        make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
    reconstruction_logger.log(
        make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())


# def on_start(state):
#     state['epoch'] = 327
#
# engine.hooks['on_start'] = on_start
engine.hooks['on_sample'] = on_sample
engine.hooks['on_forward'] = on_forward
engine.hooks['on_start_epoch'] = on_start_epoch
engine.hooks['on_end_epoch'] = on_end_epoch

engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)