import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from data_utils import preprocessing
from data_utils.preprocessing import FaceKeyDataset


TRAIN_TRANSFORMS = transforms.Compose(
    [preprocessing.PerPixelNorm(),
     preprocessing.HorizontalFlip(),  # Randomly flips
     preprocessing.Rotate(1),  # Too much rotation causes model to train badly
     preprocessing.OutputScale(),
     preprocessing.CustomToTensor()]
)

TEST_TRANSFORMS = transforms.Compose(
    [preprocessing.PerPixelNorm(),
     preprocessing.OutputScale(),
     preprocessing.CustomToTensor()]
)


# train_set = FaceKeyDataset('data/final_train.csv', transform=TRAIN_TRANSFORMS)
# train_loader = DataLoader(train_set, batch_size=8, shuffle=True)


def start_training(train_path,
                   model,
                   crit,
                   opt,
                   learn_rate,
                   mmt,
                   num_epochs,
                   batch_size,
                   val_path=None):
    train_set = FaceKeyDataset(train_path, transform=TRAIN_TRANSFORMS)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_loss = 'NA'
    if isinstance(val_path, str):
        val_set = FaceKeyDataset(val_path, transform=TEST_TRANSFORMS)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        val_loss = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    crit_dict = {
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss()
    }
    opt_dict = {
        'adam': optim.Adam(model.parameters(), lr=learn_rate),
        'adagrad': optim.Adagrad(model.parameters(), lr=learn_rate, lr_decay=0.00001),
        'sgd': optim.SGD(model.parameters(), lr=learn_rate, momentum=mmt)
    }

    criterion = crit_dict[crit]
    optimizer = opt_dict[opt]
    pbar = tqdm(range(num_epochs))

    for epoch in pbar:
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs = data['image'].to(device, dtype=torch.float)
            labels = data['keypoints'].to(device, dtype=torch.float)
            model.train()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if isinstance(val_path, str):
            model.eval()
            val_loss = start_evaluation(val_path, model, crit, batch_size)

        pbar.set_description(
            'Epoch: {0}, Train Loss: {1:4f}, Val Loss: {2:4f}'.format(epoch,
                                                                      running_loss,
                                                                      val_loss
                                                                      )
        )


def start_evaluation(val_path, model, crit, batch_size):
    test_set = FaceKeyDataset(val_path, transform=TEST_TRANSFORMS)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_loss = []
    crit_dict = {
        'mse': nn.MSELoss()
    }
    criterion = crit_dict[crit]
    with torch.no_grad():
        for data in test_loader:
            images = data['image'].to(device, dtype=torch.float)
            labels = data['keypoints'].to(device, dtype=torch.float)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss.append(loss.item())

    mean_loss = np.mean(total_loss)
    return mean_loss


def save_model(model, save_path='pretraineds/trained.pth'):
    torch.save(model.state_dict(), save_path)
    print('Model saved at: {0}'.format(save_path))


