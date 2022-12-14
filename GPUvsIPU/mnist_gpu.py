import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

 # A helper block to build convolution-pool-relu blocks.
class Block(nn.Module):
     def __init__(self, in_channels, num_filters, kernel_size, pool_size):
         super(Block, self).__init__()
         self.conv = nn.Conv2d(in_channels,
                               num_filters,
                               kernel_size=kernel_size)
         self.pool = nn.MaxPool2d(kernel_size=pool_size)
         self.relu = nn.ReLU()

     def forward(self, x):
         x = self.conv(x)
         x = self.pool(x)
         x = self.relu(x)
         return x

 # Define the network using the above blocks.
class Network(nn.Module):
     def __init__(self):
         super().__init__()
         self.layer1 = Block(1, 10, 5, 2)
         self.layer2 = Block(10, 20, 5, 2)
         self.layer3 = nn.Linear(320, 256)
         self.layer3_act = nn.ReLU()
         self.layer4 = nn.Linear(256, 10)

         self.softmax = nn.LogSoftmax(1)
         self.loss = nn.NLLLoss(reduction="mean")

     def forward(self, x, target=None):
         x = self.layer1(x)
         x = self.layer2(x)
         x = x.view(-1, 320)

         x = self.layer3_act(self.layer3(x))
         x = self.layer4(x)
         x = self.softmax(x)

         return x

def checked_gpu() :
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train(model, device, train_loader, optimizer, epoch) :
    model.train()
    pbar = tqdm(train_loader)
    pbar.set_description(f'Epoch {epoch} Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            pbar.set_postfix({'train epoch :': epoch, 'loss :' : loss.item()})

def validation(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pbar = tqdm(test_loader)
    pbar.set_description(f'validation')
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main() :
    epochs = 1
    train_mini_batch_size = 32
    validation_mini_batch_size = 16

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_data = datasets.MNIST('data', train=True, download=True,transform=transform)
    validation_data = datasets.MNIST('data', train=False, transform=transform)

    print(f'?????? ????????? ????????? : {len(train_data)}')
    print(f'?????? ????????? ????????? : {len(validation_data)}')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_mini_batch_size)
    test_loader = torch.utils.data.DataLoader(validation_data, batch_size=validation_mini_batch_size)

    print(f'torch train dataloader ????????? ????????? : {len(train_loader)}')
    print(f'torch test dataloader ????????? ????????? : {len(test_loader)}')


    train_features, _ = next(iter(train_loader))
    print(f'?????? ????????? ?????? shape : {train_features.shape}')

    device = checked_gpu()
    print(device)
    model = Network().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        validation(model, device, test_loader)

if __name__ == '__main__' :
    main()
