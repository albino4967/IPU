import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import poptorch

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

         if target is not None:
             loss = self.loss(x, target)
             return x, loss
         return x

def checked_gpu() :
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train(model, train_loader, epoch) :
    pbar = tqdm(train_loader)
    pbar.set_description(f'Epoch {epoch} Training')
    for batch_idx, (data, target) in enumerate(pbar):
        # data, target = data.to(device), target.to(device)
        # optimizer.zero_grad()
        _, loss = model(data, target)
        # loss = F.nll_loss(output, target)
        # loss.backward()
        # optimizer.step()
        if batch_idx % 10 == 0:
            pbar.set_postfix({'train epoch :': epoch, 'loss :' : loss.item()})

def validation(model, test_loader):
    test_loss = 0
    correct = 0
    pbar = tqdm(test_loader)
    pbar.set_description(f'validation')
    with torch.no_grad():
        for data, target in pbar:
            # data, target = data.to(device), target.to(device)
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

    opts = poptorch.Options()
    opts.deviceIterations(1)
    opts.Training.gradientAccumulation(64)
    opts.replicationFactor(1)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_data = datasets.MNIST('data', train=True, download=True,transform=transform)
    validation_data = datasets.MNIST('data', train=False, transform=transform)

    print(f'학습 데이터 사이즈 : {len(train_data)}')
    print(f'검증 데이터 사이즈 : {len(validation_data)}')

    ipu_training_data = poptorch.DataLoader(options=opts,
                                    dataset=train_data,
                                    batch_size=train_mini_batch_size,
                                    shuffle=True,
                                    drop_last=True)
    
    ipu_validation_data = poptorch.DataLoader(options=opts,
                                    dataset=validation_data,
                                    batch_size=validation_mini_batch_size,
                                    shuffle=True,
                                    drop_last=True)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_mini_batch_size)
    # test_loader = torch.utils.data.DataLoader(validation_data, batch_size=validation_mini_batch_size)

    print(f'IPU torch train dataloader 데이터 사이즈 : {len(ipu_training_data)}')
    print(f'IPU torch test dataloader 데이터 사이즈 : {len(ipu_validation_data)}')

    train_features, _ = next(iter(ipu_training_data))
    print(f'학습 데이터 배치 shape : {train_features.shape}')

    device = checked_gpu()
    model = Network() #.to(device=device)
    optimizer = poptorch.optim.Adam(model.parameters(), lr=0.001)

    poptorch_model = poptorch.trainingModel(model,
                                        options=opts,
                                        optimizer=optimizer)



    infer_opts = poptorch.Options()
    infer_opts.deviceIterations(1024) 
    inference_model = poptorch.inferenceModel(model, options=infer_opts)
    for epoch in range(1, epochs + 1):
        model.train()
        train(poptorch_model, ipu_training_data, epoch)
        poptorch_model.detachFromDevice()

        model.eval()
        validation(inference_model, ipu_validation_data)
        inference_model.detachFromDevice()

if __name__ == '__main__' :
    main()
