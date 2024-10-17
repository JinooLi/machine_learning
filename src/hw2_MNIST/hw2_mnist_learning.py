import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os

# 데이터셋을 다운로드할 경로 설정
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# use nvidia gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print(device)

# 데이터 전처리 정의
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 이미지를 (0,1) 범위의 텐서로 변환
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# 학습 및 테스트 데이터셋 다운로드
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# 데이터셋을 batch 단위로 불러올 수 있는 DataLoader 객체 생성.
# DataLoader는 데이터셋을 배치 단위로 분할하고, 데이터를 섞어준다.
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(
    f"예시 이미지 배치 크기: {example_data.size()}"
)  # 예: torch.Size([64, 1, 28, 28])
print(f"예시 라벨: {example_targets}")


class MnistTrain(nn.Module):
    def __init__(self):
        super(MnistTrain, self).__init__()
        # cnn
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = nn.Dropout2d(p=0.25)
        # fc
        self.layer5 = nn.Linear(64 * 14 * 14, 128)
        self.layer6 = nn.Dropout(p=0.5)
        self.layer7 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.layer5(x))
        x = torch.relu(self.layer6(x))
        x = torch.softmax(self.layer7(x), dim=1)
        return x


model = MnistTrain().to(device)
# use cross entropy loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
history = []
for epoch in range(100):
    model.train()
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        target = torch.nn.functional.one_hot(target, num_classes=10)
        target = target.type(torch.float32)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        numpy_output = output.detach().cpu().numpy()
        numpy_target = target.detach().cpu().numpy()
        numpy_output = numpy_output.argmax(axis=1)
        numpy_target = numpy_target.argmax(axis=1)
        correct_count = (numpy_output == numpy_target).sum()
        accuracy = correct_count / len(numpy_output)
        print(f"Accuracy: {accuracy}")

    history.append(loss.item())
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("loss.png")
    plt.close()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
