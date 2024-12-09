import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from final import Generator, Discriminator
import os


def load_model(model_path):
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 초기화
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 모델 가중치 로드
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])

    # 평가 모드로 설정
    generator.eval()

    return generator, device


def process_image(image_path, generator, device):
    # 이미지 전처리를 위한 transform 정의
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # 이미지 로드 및 전처리
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # 이미지 생성
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    return input_tensor, output_tensor


def save_comparison(input_tensor, output_tensor, save_path="comparison.png"):
    # [-1, 1] 범위의 텐서를 [0, 1] 범위로 변환
    def denormalize(tensor):
        return (tensor + 1) / 2

    # 입력 이미지와 출력 이미지를 가로로 나란히 배치
    comparison = torch.cat(
        [denormalize(input_tensor), denormalize(output_tensor)], dim=3
    )
    save_image(comparison, save_path)

    # matplotlib으로 이미지 표시
    plt.figure(figsize=(12, 6))
    img = plt.imread(save_path)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Input Image (left) vs Generated Image (right)")
    plt.show()


def main():
    print("테스트 모델 실행")
    # 모델 경로 설정 (이미 학습된 모델 경로)
    model_path = "./final_model/model_epoch_100.pth"
    print(f"모델 경로: {model_path}")

    # 테스트할 이미지 경로 설정
    test_image_path = "./data/test/class1/0006.png"
    print(f"테스트 이미지 경로: {test_image_path}")

    # 결과 저장 디렉토리 생성
    os.makedirs("./results", exist_ok=True)
    print(f"결과 저장 디렉토리: ./results")

    try:
        # 모델 로드
        generator, device = load_model(model_path)
        print(f"모델을 {device}에 로드했습니다.")

        # 이미지 처리
        input_tensor, output_tensor = process_image(test_image_path, generator, device)
        print("이미지 처리가 완료되었습니다.")

        # 결과 저장 및 표시
        save_comparison(input_tensor, output_tensor, "./results/comparison.png")
        print("결과가 ./results/comparison.png에 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: 모델 파일을 찾을 수 없습니다. ({model_path})")
        print("먼저 final.py를 실행하여 모델을 학습시켜주세요.")
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
