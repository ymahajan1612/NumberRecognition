import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid



transform = transforms.ToTensor()

train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)



class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)



model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []
def train_model():
	for i in range(epochs):
		trn_corr = 0


		for batch ,(X_train, y_train) in enumerate(train_loader):
			batch+=1

			# Apply the model
			y_pred = model(X_train)
			loss = criterion(y_pred, y_train)

			# Tally the number of correct predictions
			predicted = torch.max(y_pred.data, 1)[1]
			batch_corr = (predicted == y_train).sum()
			trn_corr += batch_corr

			# Update parameters
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch % 600 == 0:
				print(f'epoch: {i:2}  loss: {loss.item():10.8f}  \ accuracy: {trn_corr.item() * 100 / (10 * batch):7.3f}%')

		train_correct.append(trn_corr)
		torch.save(model.state_dict(), 'NumberGuesser.pt')

def test_model():
	test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)
	with torch.no_grad():
		correct = 0
		for X_test, y_test in test_load_all:
			y_val = model(X_test)  # we don't flatten the data this time
			predicted = torch.max(y_val, 1)[1]
			correct += (predicted == y_test).sum()
	print("Test accuracy: {}%".format((correct.item()*100)/len(test_data)))


model = ConvNet()
model.load_state_dict(torch.load('NumberGuesser.pt'))
model.eval()

