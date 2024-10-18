import torch
import torch.nn as nn

# use nvidia gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input data
input_data = torch.tensor([[0, 0], 
                           [0, 1], 
                           [1, 0], 
                           [1, 1]], dtype=torch.float32).to(device)

# output data for learning XOR
output_data = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(device)

class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x
    
model = XOR().to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05) 
print('init weight')
print(repr(model.layer1.weight))
print(repr(model.layer2.weight))
print('init result')
print(model(input_data))

total_epoch = 10000
for epoch in range(total_epoch):
    output = model(input_data)
    loss = criterion(output, output_data) # calculate loss
    optimizer.zero_grad() # initialize gradient
    loss.backward() # backpropagation
    optimizer.step() # update weights
    
    if epoch % 1000 == 999:
        print('epoch : ', epoch, '/',total_epoch , 'loss : ', loss.item())

with torch.no_grad():
    print('final result')
    print(model(input_data))
    print('final weight')
    print(repr(model.layer1.weight))
    print(repr(model.layer2.weight))