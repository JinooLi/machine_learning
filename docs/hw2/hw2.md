# Homework 2

## 목차
1. [실험 목표](##실험-목표)
2. [MNIST 분류기 구성](##MNIST-분류기-구성)
    1. [환경](###환경)
    2. [코드](###코드)
3. [실험](##실험)
    1. [실험 조건](###실험-조건)
    2. [실험 결과](###실험-결과)
    3. [결과 분석](###결과-분석)

## 실험 목표
1. MNIST DB를 분류하는 분류기를 구현하고, 
2. Layer 구조와 활성화 함수, 학습률, batch size, optimizer 종류, epoch 수 등을 다양하게 적용하여 실험을 진행한다.   
3. 이후, 각각의 조건에서의 결과들을 비교하고, 그런 결과가 나온 이유를 분석한다.

## MNIST 분류기 구성 

### 환경
- PC
    - CPU: i5-12400F
    - RAM: 32GB
    - GPU: RTX 3070ti
- Docker(WSL2)
- VSCode
    - dev container
- Python
    - Pytorch
    - Torchvision
    - Numpy
    - Matplotlib

### 코드
깃허브 링크 : [https://github.com/JinooLi/machine_learning](https://github.com/JinooLi/machine_learning)


#### 1. 필요한 라이브러리 임포트

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import os
```

---

#### 2. 작업 경로 설정 및 디바이스 선택

```python
os.chdir(os.path.dirname(os.path.realpath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("This computer uses", device)
```

- 현재 **스크립트 위치**로 작업 경로를 설정.
- **CUDA**가 가능할 경우 GPU를, 그렇지 않으면 CPU를 사용.

---

#### 3. 사용자 정의 색 반전 클래스 정의

```python
class RandomInvertColor(object):
    def __call__(self, tensor):
        if torch.rand(1) > 0.5:
            return tensor
        return 1 - tensor  # 색상 반전
```

- 입력 텐서(이미지)를 50% 확률로 **색 반전**시킨다.  
이는 데이터 **증강**(augmentation)을 위한 것이다.

---

#### 4. 데이터 전처리 파이프라인 설정

```python
transform = transforms.Compose([
    transforms.ToTensor(),  
    RandomInvertColor(),  
    transforms.Normalize(mean=(0.5,), std=(0.5,))  
])
```

- **`ToTensor()`**: 이미지를 (0,1) 범위의 텐서로 변환.
- **`RandomInvertColor()`**: 색을 랜덤으로 반전.
- **정규화**: 평균 0.5, 표준편차 0.5로 이미지 정규화.  
    원래 MNIST의 평균과 표준편차는 0.1307, 0.3081이지만, 50% 확률로 색 반전을 수행하므로 이 값은 변경됨.

---

#### 5. 데이터셋 및 데이터 로더 설정

```python
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
```

- **MNIST 데이터셋**을 다운로드하고 전처리한다.
- **DataLoader**로 데이터를 **배치(batch)** 단위로 로드.

---

#### 6. 데이터셋 예시 확인

```python
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(f"예시 이미지 배치 크기: {example_data.size()}")
print(f"예시 라벨: {example_targets}")
```

- 첫 번째 배치 데이터를 로드하여 **이미지 크기**와 **라벨** 출력.

---

#### 7. CNN 모델 정의

```python
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
```

- **CNN 네트워크** 정의.
- 여기에선 Convolutional, MaxPooling, Dropout 및 Fully Connected 레이어를 포함한다.

---

#### 8. 손실 함수 및 옵티마이저 설정

```python
model = MnistTrain().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

- loss 함수와 옵티마이저를 설정한다.
- 기본적으로 **크로스 엔트로피 손실**과 **Adam 옵티마이저**를 사용한다.

---

#### 9. 학습 루프 정의

```python
history = []
accuracy_mean = []
history_test = []
test_accuracy_mean = []
for epoch in range(50):
    model.train()
    total, total_loss, total_accuracy = 0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        target = torch.nn.functional.one_hot(target, num_classes=10).type(torch.float32)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        accuracy = (output.argmax(1) == target.argmax(1)).float().mean().item()
        total_accuracy += accuracy
        total += 1

    history.append(total_loss / total)
    accuracy_mean.append(total_accuracy / total)
```

- **50번의 epoch** 동안 학습.
- **Gradient 초기화**, **역전파**, **가중치 업데이트**를 수행.
- 각 batch의 **손실과 정확도**를 기록.

---

#### 10. 테스트 루프 정의

```python
model.eval()
test_loss, total, total_accuracy = 0, 0, 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        target = torch.nn.functional.one_hot(target, num_classes=10).type(torch.float32)

        output = model(data)
        test_loss += criterion(output, target).item()
        accuracy = (output.argmax(1) == target.argmax(1)).float().mean().item()
        total_accuracy += accuracy
        total += 1

    history_test.append(test_loss / total)
    test_accuracy_mean.append(total_accuracy / total)
```

- **학습이 끝난 후** 테스트 데이터로 평가
- **기록된 손실과 정확도**를 저장

---

#### 11. 결과 시각화

```python
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(np.linspace(1, len(history), len(history)), history)
plt.plot(np.linspace(1, len(history), len(history)), history_test)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "test"])

plt.subplot(1, 2, 2)
plt.plot(np.linspace(1, len(accuracy_mean), len(accuracy_mean)), accuracy_mean)
plt.plot(np.linspace(1, len(accuracy_mean), len(accuracy_mean)), test_accuracy_mean)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "test"])

plt.savefig("Loss-Accuracy.png", dpi=100)
plt.close()
```

- **손실과 정확도**의 변화를 그래프로 시각화 및 저장.

---

#### 12. 모델 가중치 저장 여부 확인

```python
while True:
    k = input("모델 가중치를 저장하시겠습니까? (y/n): ")
    if k == "y":
        torch.save(model.state_dict(), "model_weights.pth")
        print("모델 가중치를 저장했습니다.")
        break
    elif k == "n":
        print("모델 가중치를 저장하지 않았습니다.")
        break
    else:
        print("잘못된 입력입니다. 다시 입력해주세요.")
```

- **모델 가중치 저장 여부**를 묻고, 응답에 따라 가중치를 저장한다.



## 실험

### 실험 조건

### 실험 결과

### 결과 분석
