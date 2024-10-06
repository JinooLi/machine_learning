from dataclasses import dataclass
from enum import Enum

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

# traning data


@dataclass
class trainingData:
    data: np.array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    label: np.array = np.array([0, 1, 1, 1])

    def set_training_data(self, data: np.array, label: np.array) -> None:
        if len(data) != len(label):
            raise ValueError("data and label must have same length")
        self.data = data
        self.label = label
        print("data :", self.data)
        print("label :", self.label)


def make_data(size: int = 100) -> tuple[np.array, np.array]:
    """학습 데이터를 생성하는 함수

    Args:
        size (int, optional): 학습 데이터 개수. Defaults to 100.

    Returns:
        tuple[np.array, np.array]: 학습 데이터와 레이블
    """
    data: np.array = np.random.rand(size, 2)
    label: np.array = np.zeros(size)
    for i in range(size):
        if data[i][0] + data[i][1] > 1:
            label[i] = 0
        else:
            label[i] = 1
    return data, label


class activationType(Enum):
    sign = 0
    sigmoid = 1


class activationFunction:
    def __init__(self, activation_type: activationType = activationType.sign):
        self.activation_type = activation_type

    def fucntion(self, x: float) -> float:
        if self.activation_type == activationType.sign:
            return self._sign(x)
        elif self.activation_type == activationType.sigmoid:
            return self._sigmoid(x)
        else:
            raise ValueError("activation type is not defined")

    def diff(self, x: float) -> float:
        if self.activation_type == activationType.sign:
            return self._sign_diff(x)
        elif self.activation_type == activationType.sigmoid:
            return self._sigmoid_diff(x)
        else:
            raise ValueError("activation type is not defined")

    def _sign(self, x: float) -> int:
        if x > 0:
            return 1
        else:
            return 0

    def _sign_diff(self, x: int) -> int:
        return 1

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def _sigmoid_diff(self, x: float) -> float:
        return x * (1 - x)


# perceptron
def perceptron(
    data: np.array,
    weights: np.array,
    activation_type: activationType = activationType.sign,
) -> int:
    # If data's lenght and weights's length are different, raise error
    if len(data) != len(weights):
        raise ValueError("data and weights must have same length")

    # calculate dot product
    dot = data @ weights

    activation = activationFunction(activation_type)

    # use sign function
    return activation.fucntion(dot)


# training perceptron


def train_perceptron(
    data: np.array,
    label: np.array,
    weights: np.array,
    epochs: int = 100,
    learning_rate: float = 0.05,
    activation_type: activationType = activationType.sign,
) -> np.array:
    activation = activationFunction(activation_type)
    for trial in range(epochs):
        update_weights = np.zeros(len(weights))
        for i in range(len(data)):
            # add bias
            data_with_bias = np.concatenate((data[i], [1]))
            # run perceptron
            perceptron_output: float = perceptron(
                data_with_bias, weights, activation_type
            )
            # if the output is not correct, update weights
            error = label[i] - perceptron_output
            update_weights += (
                activation.diff(perceptron_output) * data_with_bias * error
            )
        # update weights
        weights += learning_rate * update_weights

        # print & plot weights every epochs/7
        if trial % (epochs // 7) == 0:
            print("epoch :", trial + 1, " weights :", weights)
            plt.subplot(2, 5, 2 + trial // (epochs // 7))
            plt.title("epoch : " + str(trial + 1))
            plt.scatter(data[:, 0], data[:, 1], c=label)
            x = np.linspace(-0.1, 1.1, 10)
            y = (-weights[0] * x - weights[2]) / weights[1]
            plt.plot(x, y)

    return weights


def main():
    # make training data
    training_data: trainingData = trainingData()
    data, label = make_data(80)
    training_data.set_training_data(data, label)

    # init weights
    weights = np.random.rand(3)
    print("init weigts : ", weights)

    # plot before training
    plt.subplot(2, 5, 1)
    plt.title("Before training")
    plt.scatter(
        training_data.data[:, 0], training_data.data[:, 1], c=training_data.label
    )
    x = np.linspace(-0.1, 1.1, 10)
    y = (-weights[0] * x - weights[2]) / weights[1]
    plt.plot(x, y)

    # train perceptron
    weights = train_perceptron(
        training_data.data,
        training_data.label,
        weights,
        epochs=120,
        learning_rate=0.05,
        activation_type=activationType.sigmoid,
    )
    print("trained weigts", weights)

    # plot after training
    plt.subplot(2, 5, 10)
    plt.title("After training")
    plt.scatter(
        training_data.data[:, 0], training_data.data[:, 1], c=training_data.label
    )
    y = (-weights[0] * x - weights[2]) / weights[1]
    plt.plot(x, y)

    plt.show()


if __name__ == "__main__":
    main()
