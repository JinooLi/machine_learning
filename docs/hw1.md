# Homework 1

## 작업 환경
- OS: Ubuntu 22.04.1 LTS
- conda 24.7.1 (Python 3.10.14)
- conda environment
    ```yaml
    name: ml
    channels:
    - defaults
    dependencies:
    - _libgcc_mutex=0.1=main
    - _openmp_mutex=5.1=1_gnu
    - blas=1.0=mkl
    - bottleneck=1.3.7=py310ha9d4c09_0
    - brotli=1.0.9=h5eee18b_8
    - brotli-bin=1.0.9=h5eee18b_8
    - bzip2=1.0.8=h5eee18b_6
    - ca-certificates=2024.7.2=h06a4308_0
    - contourpy=1.2.0=py310hdb19cb5_0
    - cycler=0.11.0=pyhd3eb1b0_0
    - cyrus-sasl=2.1.28=h52b45da_1
    - dbus=1.13.18=hb2f20db_0
    - expat=2.6.2=h6a678d5_0
    - fontconfig=2.14.1=h55d465d_3
    - fonttools=4.51.0=py310h5eee18b_0
    - freetype=2.12.1=h4a9f257_0
    - glib=2.78.4=h6a678d5_0
    - glib-tools=2.78.4=h6a678d5_0
    - gst-plugins-base=1.14.1=h6a678d5_1
    - gstreamer=1.14.1=h5eee18b_1
    - icu=73.1=h6a678d5_0
    - intel-openmp=2023.1.0=hdb19cb5_46306
    - jpeg=9e=h5eee18b_3
    - kiwisolver=1.4.4=py310h6a678d5_0
    - krb5=1.20.1=h143b758_1
    - lcms2=2.12=h3be6417_0
    - ld_impl_linux-64=2.38=h1181459_1
    - lerc=3.0=h295c915_0
    - libbrotlicommon=1.0.9=h5eee18b_8
    - libbrotlidec=1.0.9=h5eee18b_8
    - libbrotlienc=1.0.9=h5eee18b_8
    - libclang=14.0.6=default_hc6dbbc7_1
    - libclang13=14.0.6=default_he11475f_1
    - libcups=2.4.2=h2d74bed_1
    - libdeflate=1.17=h5eee18b_1
    - libedit=3.1.20230828=h5eee18b_0
    - libffi=3.4.4=h6a678d5_1
    - libgcc-ng=11.2.0=h1234567_1
    - libglib=2.78.4=hdc74915_0
    - libgomp=11.2.0=h1234567_1
    - libiconv=1.16=h5eee18b_3
    - libllvm14=14.0.6=hdb19cb5_3
    - libpng=1.6.39=h5eee18b_0
    - libpq=12.17=hdbd6064_0
    - libstdcxx-ng=11.2.0=h1234567_1
    - libtiff=4.5.1=h6a678d5_0
    - libuuid=1.41.5=h5eee18b_0
    - libwebp-base=1.3.2=h5eee18b_0
    - libxcb=1.15=h7f8727e_0
    - libxkbcommon=1.0.1=h097e994_2
    - libxml2=2.13.1=hfdd30dd_2
    - lz4-c=1.9.4=h6a678d5_1
    - matplotlib=3.9.2=py310h06a4308_0
    - matplotlib-base=3.9.2=py310hbfdbfaf_0
    - mkl=2023.1.0=h213fc3f_46344
    - mkl-service=2.4.0=py310h5eee18b_1
    - mkl_fft=1.3.10=py310h5eee18b_0
    - mkl_random=1.2.7=py310h1128e8f_0
    - mysql=5.7.24=h721c034_2
    - ncurses=6.4=h6a678d5_0
    - numexpr=2.8.7=py310h85018f9_0
    - numpy=1.26.4=py310h5f9d8c6_0
    - numpy-base=1.26.4=py310hb5e798b_0
    - openjpeg=2.5.2=he7f1fd0_0
    - openssl=3.0.14=h5eee18b_0
    - packaging=24.1=py310h06a4308_0
    - pandas=2.2.2=py310h6a678d5_0
    - pcre2=10.42=hebb0a14_1
    - pillow=10.4.0=py310h5eee18b_0
    - pip=24.2=py310h06a4308_0
    - ply=3.11=py310h06a4308_0
    - pyparsing=3.1.2=py310h06a4308_0
    - python=3.10.14=h955ad1f_1
    - python-dateutil=2.9.0post0=py310h06a4308_2
    - python-tzdata=2023.3=pyhd3eb1b0_0
    - pytz=2024.1=py310h06a4308_0
    - qt-main=5.15.2=h53bd1ea_10
    - readline=8.2=h5eee18b_0
    - setuptools=72.1.0=py310h06a4308_0
    - sip=6.7.12=py310h6a678d5_0
    - six=1.16.0=pyhd3eb1b0_1
    - sqlite=3.45.3=h5eee18b_0
    - tbb=2021.8.0=hdb19cb5_0
    - tk=8.6.14=h39e8969_0
    - tomli=2.0.1=py310h06a4308_0
    - tornado=6.4.1=py310h5eee18b_0
    - tzdata=2024a=h04d1e81_0
    - unicodedata2=15.1.0=py310h5eee18b_0
    - wheel=0.43.0=py310h06a4308_0
    - xz=5.4.6=h5eee18b_1
    - zlib=1.2.13=h5eee18b_1
    - zstd=1.5.5=hc292b87_2
    - pip:
        - anyio==4.4.0
        - argon2-cffi==23.1.0
        - argon2-cffi-bindings==21.2.0
        - arrow==1.3.0
        - asttokens==2.4.1
        - async-lru==2.0.4
        - attrs==24.2.0
        - babel==2.16.0
        - beautifulsoup4==4.12.3
        - bleach==6.1.0
        - certifi==2024.8.30
        - cffi==1.17.0
        - charset-normalizer==3.3.2
        - click==8.1.7
        - comm==0.2.2
        - debugpy==1.8.5
        - decorator==5.1.1
        - defusedxml==0.7.1
        - exceptiongroup==1.2.2
        - executing==2.1.0
        - fastjsonschema==2.20.0
        - filelock==3.15.4
        - fqdn==1.5.1
        - fsspec==2024.6.1
        - h11==0.14.0
        - httpcore==1.0.5
        - httpx==0.27.2
        - idna==3.8
        - ipykernel==6.29.5
        - ipython==8.27.0
        - isoduration==20.11.0
        - jedi==0.19.1
        - jinja2==3.1.4
        - json5==0.9.25
        - jsonpointer==3.0.0
        - jsonschema==4.23.0
        - jsonschema-specifications==2023.12.1
        - jupyter-client==8.6.2
        - jupyter-core==5.7.2
        - jupyter-events==0.10.0
        - jupyter-lsp==2.2.5
        - jupyter-server==2.14.2
        - jupyter-server-terminals==0.5.3
        - jupyterlab==4.2.5
        - jupyterlab-pygments==0.3.0
        - jupyterlab-server==2.27.3
        - markupsafe==2.1.5
        - matplotlib-inline==0.1.7
        - mistune==3.0.2
        - mpmath==1.3.0
        - nbclient==0.10.0
        - nbconvert==7.16.4
        - nbformat==5.10.4
        - nest-asyncio==1.6.0
        - networkx==3.3
        - notebook==7.2.2
        - notebook-shim==0.2.4
        - nvidia-cublas-cu12==12.1.3.1
        - nvidia-cuda-cupti-cu12==12.1.105
        - nvidia-cuda-nvrtc-cu12==12.1.105
        - nvidia-cuda-runtime-cu12==12.1.105
        - nvidia-cudnn-cu12==9.1.0.70
        - nvidia-cufft-cu12==11.0.2.54
        - nvidia-curand-cu12==10.3.2.106
        - nvidia-cusolver-cu12==11.4.5.107
        - nvidia-cusparse-cu12==12.1.0.106
        - nvidia-nccl-cu12==2.20.5
        - nvidia-nvjitlink-cu12==12.6.68
        - nvidia-nvtx-cu12==12.1.105
        - overrides==7.7.0
        - pandocfilters==1.5.1
        - parso==0.8.4
        - pexpect==4.9.0
        - platformdirs==4.2.2
        - prometheus-client==0.20.0
        - prompt-toolkit==3.0.47
        - psutil==6.0.0
        - ptyprocess==0.7.0
        - pure-eval==0.2.3
        - pycparser==2.22
        - pygments==2.18.0
        - pyqt5==5.15.11
        - pyqt5-qt5==5.15.15
        - pyqt5-sip==12.15.0
        - python-json-logger==2.0.7
        - pyyaml==6.0.2
        - pyzmq==26.2.0
        - qt5-applications==5.15.2.2.3
        - qt5-tools==5.15.2.1.3
        - referencing==0.35.1
        - requests==2.32.3
        - rfc3339-validator==0.1.4
        - rfc3986-validator==0.1.1
        - rpds-py==0.20.0
        - send2trash==1.8.3
        - sniffio==1.3.1
        - soupsieve==2.6
        - stack-data==0.6.3
        - sympy==1.13.2
        - terminado==0.18.1
        - tinycss2==1.3.0
        - torch==2.4.0
        - traitlets==5.14.3
        - triton==3.0.0
        - types-python-dateutil==2.9.0.20240821
        - typing-extensions==4.12.2
        - uri-template==1.3.0
        - urllib3==2.2.2
        - wcwidth==0.2.13
        - webcolors==24.8.0
        - webencodings==0.5.1
        - websocket-client==1.8.0
    prefix: /home/jinoo/anaconda3/envs/ml
    ```
    
    이 파일을 `ml.yml`로 저장하고 다음 명령어로 가상환경을 생성한다.

    ```bash
    conda env create -f ml.yaml
    ```

    가상환경을 활성화한다.

    ```bash
    conda activate ml
    ```

    이후 작업을 진행한다.

## 목표
데이터 집합에서 두 개의 cluster of points를 퍼셉트론 학습으로 분리한다. 이때 반드시 퍼셉트론 구조로 프로그래밍해야 하며, 결정 경계를 시각화해야 한다.


### 데이터 집합
데이터 집합은 2차원 평면상에서 x좌표와 y좌표가 0과 1사이의 랜덤값을 가지는 점 80개로 구성하였다.   
두 개의 cluster는 직선 $y-x-1=0$을 기준으로 나뉘어 있다.  
lable은 $y-x-1>0$이면 1, $y-x-1<0$이면 0으로 설정하였다.

```python
# traning data
@dataclass
class trainingData:
    data : np.array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    label : np.array = np.array([0, 1, 1, 1])

    def set_training_data(self, data:np.array, label:np.array)->None:
        if len(data) != len(label):
            raise ValueError('data and label must have same length')
        self.data = data
        self.label = label
        print('data :', self.data)
        print('label :', self.label)

def make_data(size:int=100)->tuple[np.array, np.array]:
    """학습 데이터를 생성하는 함수

    Args:
        size (int, optional): 학습 데이터 개수. Defaults to 100.

    Returns:
        tuple[np.array, np.array]: 학습 데이터와 레이블
    """
    data:np.array = np.random.rand(size, 2)
    label:np.array = np.zeros(size)
    for i in range(size):
        if data[i][0] + data[i][1] > 1:
            label[i] = 0
        else:
            label[i] = 1
    return data, label
```

### 퍼셉트론 설계
퍼셉트론의 활성함수로 sign 함수와 sigmoid 함수를 고를 수 있도록 하였다.   
시그모이드 함수의 출력에 맞추기 위해 sign 함수는 0보다 크면 1, 작으면 0을 반환하도록 하였다. 사실상 unit step function과 같다.
또한, 역전파를 위해 활성함수의 미분도 구현하였다. 그러나 sign 함수의 미분은 학습을 위해 모든 구간에서 1로 설정하였다.

```python
from enum import Enum
class activationType(Enum):
    sign = 0
    sigmoid = 1

class activationFunction:
    def __init__(self, activation_type:activationType=activationType.sign):
        self.activation_type = activation_type

    def fucntion(self, x:float)->float:
        if self.activation_type == activationType.sign:
            return self._sign(x)
        elif self.activation_type == activationType.sigmoid:
            return self._sigmoid(x)
        else:
            raise ValueError('activation type is not defined')

    def diff(self, x:float)->float:
        if self.activation_type == activationType.sign:
            return self._sign_diff(x)
        elif self.activation_type == activationType.sigmoid:
            return self._sigmoid_diff(x)
        else:
            raise ValueError('activation type is not defined')

    def _sign(self, x:float)->int:
        if x > 0:
            return 1
        else:
            return 0
    
    def _sign_diff(self, x:int)->int:
        return 1
        
    def _sigmoid(self, x:float)->float:
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_diff(self, x:float)->float:
        return x * (1 - x)

# perceptron
def perceptron(data:np.array, weights:np.array, activation_type:activationType=activationType.sign)->int:
    # If data's lenght and weights's length are different, raise error
    if len(data) != len(weights):
        raise ValueError('data and weights must have same length')
    
    # calculate dot product
    dot = data @ weights

    activation = activationFunction(activation_type)

    # use sign function
    return activation.fucntion(dot)
```

### 퍼셉트론 학습
퍼셉트론 학습은 다음과 같이 진행한다.
1. 데이터와 초기 가중치를 입력받는다.
2. 학습률과 학습 횟수를 설정한다.
3. 학습을 진행한다.
4. 학습 중간에 가중치를 출력하고 시각화한다.
5. 학습된 가중치를 반환한다.

다음은 그 코드이다.

```python
# training perceptron
def train_perceptron(data:np.array, label:np.array, weights:np.array, epochs:int=100, learning_rate:float=0.05, activation_type:activationType=activationType.sign)->np.array:
    activation = activationFunction(activation_type)
    for trial in range(epochs):
        update_weights = np.zeros(len(weights))
        for i in range(len(data)):
            # add bias
            data_with_bias = np.concatenate((data[i], [1]))
            # run perceptron
            perceptron_output:float = perceptron(data_with_bias, weights, activation_type)
            # if the output is not correct, update weights
            error = label[i] - perceptron_output
            update_weights += activation.diff(perceptron_output) * data_with_bias * error
        # update weights
        weights += learning_rate * update_weights

        # print & plot weights every epochs/7
        if trial % (epochs // 7) == 0:
            print('epoch :', trial, ' weights :', weights)
            plt.subplot(2, 5, 2 + trial // (epochs // 7))
            plt.title('trial : ' + str(trial+1))
            plt.scatter(data[:,0], 
                        data[:,1], 
                        c=label)
            x = np.linspace(-0.1, 1.1, 10)
            y = (-weights[0]*x - weights[2])/weights[1]
            plt.plot(x, y)

    return weights
```

## 결과

main 함수에서 데이터를 생성하고 퍼셉트론 학습을 진행한다.   
처음 가중치는 랜덤값으로 설정하였다. 데이터의 개수도 80개로 설정하였다.

```python
def main():
    # make training data
    training_data:trainingData = trainingData()
    data, label = make_data(80)
    training_data.set_training_data(data, label)

    # init weights
    weights = np.random.rand(3)
    print('init weigts : ', weights)

    # plot before training
    plt.subplot(2, 5, 1)
    plt.title('Before training')
    plt.scatter(training_data.data[:,0], 
                training_data.data[:,1], 
                c=training_data.label)
    x = np.linspace(-0.1, 1.1, 10)
    y = (-weights[0]*x - weights[2])/weights[1]
    plt.plot(x, y)

    # train perceptron
    weights = train_perceptron(training_data.data, 
                               training_data.label, 
                               weights, 
                               epochs=100,
                               learning_rate=0.05,
                               activation_type=activationType.sign)
    print('trained weigts', weights)
    
    # plot after training
    plt.subplot(2, 5, 10)
    plt.title('After training')
    plt.scatter(training_data.data[:,0], training_data.data[:,1], c=training_data.label)
    y = (-weights[0]*x - weights[2])/weights[1]
    plt.plot(x, y)

    plt.show()
```

결과는 다음과 같다.

### 활성함수: sign 함수일 때

#### 터미널 출력

```bash
init weigts :  [0.76972117 0.6395787  0.25109306]
epoch : 1  weights : [-0.54841381 -0.57436903 -1.64890694]
epoch : 18  weights : [-2.37789589 -1.90940262  3.15109306]
epoch : 35  weights : [-3.5562628  -3.03192232  2.90109306]
epoch : 52  weights : [-3.39100495 -3.12899759  3.80109306]
epoch : 69  weights : [-3.83898259 -3.75424185  3.40109306]
epoch : 86  weights : [-3.75034166 -3.61487463  4.20109306]
epoch : 103  weights : [-4.21043951 -4.01717821  3.85109306]
epoch : 120  weights : [-4.15207533 -4.02985684  4.05109306]
trained weigts [-4.15207533 -4.02985684  4.05109306]
```
#### 그래프
![alt text](image.png)

### 활성함수: sigmoid 함수일 때
#### 터미널 출력
```bash
init weigts :  [0.03472196 0.58867787 0.39076105]
epoch : 1  weights : [-0.13597145  0.42213597  0.21210854]
epoch : 18  weights : [-1.06962941 -0.57966799  0.55737022]
epoch : 35  weights : [-1.58222821 -1.18573104  1.18879559]
epoch : 52  weights : [-1.99188254 -1.68219429  1.68432869]
epoch : 69  weights : [-2.33304792 -2.09588639  2.0896635 ]
epoch : 86  weights : [-2.62756628 -2.44912412  2.4333551 ]
epoch : 103  weights : [-2.88859392 -2.7576251   2.73299613]
epoch : 120  weights : [-3.12432813 -3.03220806  2.99980348]
trained weigts [-3.12432813 -3.03220806  2.99980348]
```
#### 그래프
![alt text](image-1.png)