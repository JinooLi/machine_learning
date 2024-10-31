import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import numpy as np
import os
import time

# 데이터셋을 다운로드할 경로 설정
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# use nvidia gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("This computer uses", device)


# 파라미터 클래스 정의.
# 이 클래스는 학습에 필요한 파라미터를 저장하는 클래스이다.
class params:
    def __init__(
        self,
        name: str = "MNIST",
        batch_size: int = 1024,
        lr: float = 0.001,
        epoch: int = 50,
        normalize_mean: float = 0.5,
        normalize_std: float = 0.5,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        loss: nn.modules.loss._Loss = nn.CrossEntropyLoss,
    ) -> None:
        self.name = name
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.optimizer = optimizer
        self.loss = loss


# 파라미터 객체 생성.
# 여기에서 실험에서 변경할 파라미터를 설정한다.
param = params()


# cnn 모델 정의
class MnistTrain(nn.Module):
    def __init__(self):
        super(MnistTrain, self).__init__()
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = nn.Dropout2d(p=0.25)
        self.layer5 = nn.Linear(64 * 14 * 14, 128)
        self.layer6 = nn.Dropout(p=0.5)
        self.layer7 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer5(x))
        x = torch.relu(self.layer6(x))
        x = torch.softmax(self.layer7(x), dim=1)
        return x


# 사용자 정의 색 반전 클래스 정의
# 랜덤으로 색상을 반전하거나 엣지 이미지로 변환하는 클래스
class RandomInvertColor(object):
    def __call__(self, tensor):
        rand = torch.rand(1)
        if rand > 0.66:  # 그대로
            return tensor
        elif rand > 0.33:  # 반전
            return 1 - tensor
        else:  # Edge 이미지로 변환
            tensor = (tensor > 0.5).float()
            tensor = tensor.squeeze().numpy()
            contour = np.zeros_like(tensor)
            for i in range(1, tensor.shape[0] - 1):
                for j in range(1, tensor.shape[1] - 1):
                    if tensor[i, j] == 1:
                        if (
                            tensor[i - 1, j] == 0
                            or tensor[i + 1, j] == 0
                            or tensor[i, j - 1] == 0
                            or tensor[i, j + 1] == 0
                        ):
                            contour[i, j] = 1
            return torch.tensor(contour).unsqueeze(0)


# 데이터 전처리 정의
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 이미지를 (0,1) 범위의 텐서로 변환
        RandomInvertColor(),  # 이미지의 색상을 랜덤으로 반전
        # ToContour(),  # 이미지를 엣지 이미지로 변환
        transforms.Normalize(
            mean=(param.normalize_mean,), std=(param.normalize_std,)
        ),  # 이미지를 정규화한다.
    ]
)

# 학습 및 테스트 데이터셋 다운로드
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# 이미지 출력
img_loader = DataLoader(train_dataset, batch_size=9, shuffle=True)
images, labels = next(iter(img_loader))

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle("train_dataset", fontsize=16)

for i in range(9):
    ax = axes[i // 3, i % 3]
    ax.imshow(images[i].squeeze(), cmap="gray")
    ax.set_title(f"Label: {labels[i]}")
    ax.axis("off")

plt.savefig("train_dataset.png", dpi=200)

# 모델 구조 출력
summary(MnistTrain(), (1, 28, 28))


def experiment(exp_name: str):
    # 실험 이름 설정
    param.name = exp_name
    print(param.name)

    # 데이터셋을 batch 단위로 불러올 수 있는 DataLoader 객체 생성.
    # DataLoader는 데이터셋을 배치 단위로 분할하고, 데이터를 섞어준다.
    train_loader = DataLoader(
        train_dataset, batch_size=param.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=4
    )

    # 데이터셋 확인
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(f"예시 이미지 배치 크기: {example_data.size()}")
    print(f"예시 라벨: {example_targets}")

    model = MnistTrain().to(device)

    # loss function, optimizer 정의
    criterion = param.loss()
    optimizer = param.optimizer(model.parameters(), lr=param.lr)
    print("loss function: ", criterion)
    print("optimizer: ", optimizer)

    # 학습 루프
    history = []
    accuracy_mean = []
    history_test = []
    test_accuracy_mean = []
    total_time = 0
    for _epoch in range(param.epoch):
        start_time = time.time()
        model.train()
        total = 0
        total_loss = 0
        total_accuracy = 0
        for data, target in train_loader:
            print("\rtrain: " + str(total), end="")
            data = data.to(device)
            target = target.to(device)

            # target을 one-hot encoding으로 변환
            target = torch.nn.functional.one_hot(target, num_classes=10)
            target = target.type(torch.float32)

            # 모델의 gradient를 초기화
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 모든 loss, accuracy를 더한다.
            total_loss += loss.item()

            numpy_output: np.ndarray = output.detach().cpu().numpy()
            numpy_target: np.ndarray = target.detach().cpu().numpy()
            numpy_output = numpy_output.argmax(axis=1)
            numpy_target = numpy_target.argmax(axis=1)
            correct_count = (numpy_output == numpy_target).sum()
            accuracy = correct_count / len(numpy_output)
            total_accuracy += accuracy

            total += 1

        end_time = time.time()
        total_time += end_time - start_time

        history.append(total_loss / total)
        accuracy_mean.append(total_accuracy / total)

        # 한 epoch마다 test_data로 모델 평가
        model.eval()
        test_loss = 0
        total = 0
        total_accuracy = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)

                target = torch.nn.functional.one_hot(target, num_classes=10)
                target = target.type(torch.float32)

                output = model(data)
                test_loss += criterion(output, target).item()
                total += 1

                numpy_output = output.detach().cpu().numpy()
                numpy_target = target.detach().cpu().numpy()
                numpy_output = numpy_output.argmax(axis=1)
                numpy_target = numpy_target.argmax(axis=1)
                correct_count = (numpy_output == numpy_target).sum()
                accuracy = correct_count / len(numpy_output)
                total_accuracy += accuracy

        history_test.append(test_loss / total)
        test_accuracy_mean.append(total_accuracy / total)

        print(
            f"\rEpoch: {_epoch+1:2d}, Test Loss: {history_test[-1]:.4f}, Test Accuracy: {test_accuracy_mean[-1]:.4f}, Learning Time: {end_time - start_time:.2f} sec"
        )

    plt.figure(figsize=(14, 12))
    plt.tight_layout()

    # params 내용 출력
    plt.subplot(2, 20, 1)
    plt.title(param.name)
    plt.text(0, 0.8, "batch size: " + str(param.batch_size), fontsize=12, ha="left")
    plt.text(0, 0.75, "learning rate: " + str(param.lr), fontsize=12, ha="left")
    plt.text(0, 0.7, "epoch: " + str(param.epoch), fontsize=12, ha="left")
    plt.text(
        0, 0.65, "normalize mean: " + str(param.normalize_mean), fontsize=12, ha="left"
    )
    plt.text(
        0, 0.6, "normalize std: " + str(param.normalize_std), fontsize=12, ha="left"
    )
    plt.text(0, 0.55, "optimizer: " + str(param.optimizer), fontsize=12, ha="left")
    plt.text(0, 0.5, "loss: " + str(param.loss), fontsize=12, ha="left")
    plt.text(
        0,
        0.45,
        "Total Learning Time: " + str((total_time // 0.01) / 100) + " sec",
        fontsize=12,
        ha="left",
    )
    plt.text(
        0,
        0.4,
        "Average Time per epoch: "
        + str(((total_time / param.epoch) // 0.01) / 100)
        + " sec",
        fontsize=12,
        ha="left",
    )
    plt.text(
        0,
        0.35,
        "Final Test Accuracy: " + str((test_accuracy_mean[-1] // 0.001) / 1000),
        fontsize=12,
        ha="left",
    )
    plt.text(
        0,
        0.3,
        "Final Test Loss: " + str((history_test[-1] // 0.001) / 1000),
        fontsize=12,
        ha="left",
    )
    plt.axis("off")

    # loss, accuracy 그래프 출력
    linspace = np.linspace(1, len(history), len(history))
    plt.subplot(2, 2, 3)
    plt.plot(linspace, history)
    plt.plot(linspace, history_test)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend(["train", "test"])

    plt.subplot(2, 2, 4)
    plt.plot(linspace, accuracy_mean)
    plt.plot(linspace, test_accuracy_mean)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend(["train", "test"])

    plt.savefig(param.name + "-Loss-Accuracy.png", dpi=200)
    plt.close()

    print(
        f"Total Learning Time: {total_time:.2f} sec, Average Time per epoch: {total_time / param.epoch:.2f} sec"
    )


for i in range(3):
    experiment("Exp4-" + str(i + 1))
