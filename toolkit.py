import torch
import numpy as np
from PIL import Image

from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for PyTorch model,
        returns an Numpy array
    '''
    width, height = image.size

    short = width if width < height else height
    long = height if height > width else width

    new_short, new_long = 256, int(256/short*long)

    img = image.resize((new_short, new_long))

    left, top = (new_short - 224) / 2, (new_long - 224) / 2
    area = (left, top, 224+left, 224+top)
    img_new = img.crop(area)
    np_img = np.array(img_new)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_img = (np_img / 255 - mean) / std
    image = np.transpose(np_img, (2, 0, 1))

    return image.astype(np.float32)


def get_dataloders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'training': transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]),

        'validating': transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),

        'testing': transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    }

    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validating': datasets.ImageFolder(valid_dir, transform=data_transforms['validating']),
        'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing'])
    }

    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'validating': torch.utils.data.DataLoader(image_datasets['validating'], batch_size=64, shuffle=True),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=30, shuffle=False)
    }

    class_to_idx = image_datasets['training'].class_to_idx
    return dataloaders, class_to_idx


def model_config(struc, hidden_units):
    if struc == 'vgg16':
        model = models.vgg16(pretrained=True)
        classifier_input_size = model.classifier[0].in_features
    elif struc == 'densenet121':
        model = models.densenet121(pretrained=True)
        classifier_input_size = model.classifier.in_features       
    else:
        #Drag to vgg16
        print("in this test you must choose either use: vgg16 or densenet121")
        print("what you input is incorrect, use default value: vgg16")
        model = models.vgg16(pretrained=True)
        classifier_input_size = model.classifier[0].in_features

   # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier_output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, classifier_output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


def model_create(struc, learning_rate, hidden_units, class_to_idx):
    # Load model
    model = model_config(struc, hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.zero_grad()
    # Save class to index mapping
    model.class_to_idx = class_to_idx

    return model, optimizer, criterion


def save_checkpoint(file, model, optimizer, struc, learning_rate, epochs):
    checkpoint = {'input_size': 1024,
                  'architectures': 'densenet121',
                  'learing_rate': learning_rate,
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'output_size': 102,
                  'epochs': epochs,
                  'arch': 'densenet121',
                  'state_dict': model.state_dict(),
                  }

    torch.save(checkpoint, file)


def load_checkpoint(file):
    checkpoint = torch.load(file)
    class_to_idx = checkpoint['class_to_idx']
    learning_rate = checkpoint['learing_rate']
    model, optimizer, criterion = model_create('densenet121',learning_rate, 500, class_to_idx)
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.epochs = checkpoint['epochs']

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    return model


# Do validation on the test set

def validation(model, dataloaders, criterion):
    correct = 0
    total = 0
    model.eval()  # turn off dropout
    with torch.no_grad():
        for data in dataloaders:
            images, labels = data
            gpu = torch.cuda.is_available()
            if gpu:
                images = Variable(images.float().cuda())
                labels = Variable(labels.long().cuda())
            else:
                images = Variable(images, volatile=True)
                labels = Variable(labels, volatile=True)
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def train(model, trainloader, epochs, print_every, criterion, optimizer, device='gpu'):
    #epochs = epochs
    #print_every = print_every
    steps = 0
    
    model.to('cuda')# use cuda

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0


def predict(image_path, model, gpu, topk=5):
    # Predict the class(es) of an image using a trained model.

    gpu = torch.cuda.is_available()
    image = Image.open(image_path)
    np_image = process_image(image)
    model.eval()

    tensor_image = torch.from_numpy(np_image)

    if gpu:
        tensor_image = Variable(tensor_image.float().cuda())
    else:
        tensor_image = Variable(tensor_image)

    tensor_image = tensor_image.unsqueeze(0)
    output = model.forward(tensor_image)
    ps = torch.exp(output).data.topk(topk)

    probs = ps[0].cpu() if gpu else ps[0]
    classes = ps[1].cpu() if gpu else ps[1]

    inverted_class_to_idx = {
        model.class_to_idx[c]: c for c in model.class_to_idx}

    mapped_classes = list(
        inverted_class_to_idx[label] for label in classes.numpy()[0]
        )

    return probs.numpy()[0], mapped_classes
