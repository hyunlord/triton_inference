import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import argparse

# Triton 서버 정보
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "deep_hashing"
MODEL_VERSION = "1"

# config.py에서 가져온 이미지 크기
IMAGE_SIZE = 384

# config.py에서 가져온 bit_list
BIT_LIST = [8, 16, 32, 48, 64, 128]

def preprocess_image(image_path, image_size):
    """이미지 파일을 로드하고 Triton 서버 입력 형식에 맞게 전처리합니다."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((image_size, image_size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32)

        # HWC to CHW (Height, Width, Channel -> Channel, Height, Width)
        img_np = img_np.transpose((2, 0, 1))

        # Normalize to [0, 1] if not already (assuming model expects this)
        # SigLIP models typically expect pixel values in [0, 1] or normalized.
        # If your model expects different normalization (e.g., ImageNet stats), adjust here.
        img_np = img_np / 255.0

        # Add batch dimension (1, C, H, W)
        img_np = np.expand_dims(img_np, axis=0)
        return img_np
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def calculate_hamming_distance(hash1, hash2):
    """두 이진 해시 코드 간의 해밍 거리를 계산합니다."""
    return np.sum(hash1 != hash2)

def main():
    parser = argparse.ArgumentParser(description="Triton Inference Client for Deep Hashing Model.")
    parser.add_argument("image_path1", type=str, help="Path to the first image file.")
    parser.add_argument("image_path2", type=str, help="Path to the second image file.")
    args = parser.parse_args()

    try:
        # Triton HTTP 클라이언트 생성
        triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

        # 서버 및 모델 상태 확인
        print(f"Checking server readiness: {triton_client.is_server_ready()}")
        print(f"Checking model readiness for {MODEL_NAME}: {triton_client.is_model_ready(MODEL_NAME)}")

        # 이미지 전처리
        image1_data = preprocess_image(args.image_path1, IMAGE_SIZE)
        image2_data = preprocess_image(args.image_path2, IMAGE_SIZE)

        if image1_data is None or image2_data is None:
            return

        # 두 이미지를 하나의 배치로 결합
        # Triton config.pbtxt의 max_batch_size가 1보다 크므로 가능
        batch_image_data = np.concatenate((image1_data, image2_data), axis=0)

        # 입력 객체 생성
        inputs = []
        inputs.append(httpclient.InferInput("images", batch_image_data.shape, "FP32"))
        inputs[0].set_data_from_numpy(batch_image_data, binary_data=True)

        # 출력 객체 생성 (config.pbtxt에 정의된 출력 이름 사용)
        outputs = []
        for bit in BIT_LIST:
            outputs.append(httpclient.InferRequestedOutput(f"output_{bit}_bit", binary_data=True))

        # 추론 요청 보내기
        print(f"\nSending inference request to model: {MODEL_NAME} with batch size {batch_image_data.shape[0]}")
        response = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
            model_version=MODEL_VERSION
        )

        # 결과 확인 및 해시 코드 비교
        print("\nInference Results:")
        hash_codes_128_bit = []

        for bit in BIT_LIST:
            output_name = f"output_{bit}_bit"
            output_data = response.as_numpy(output_name)

            if output_data is not None:
                # 배치에서 각 이미지의 출력 분리
                output_image1 = output_data[0]
                output_image2 = output_data[1]

                # torch.sign과 동일하게 np.sign 적용하여 이진 해시 코드 생성
                signed_hash_image1 = np.sign(output_image1)
                signed_hash_image2 = np.sign(output_image2)

                print(f"  {bit}-bit Hash Code (Image 1) shape: {signed_hash_image1.shape}")
                print(f"  {bit}-bit Hash Code (Image 2) shape: {signed_hash_image2.shape}")

                if bit == 128: # 128비트 해시 코드 저장
                    hash_codes_128_bit.append(signed_hash_image1)
                    hash_codes_128_bit.append(signed_hash_image2)
            else:
                print(f"  Output '{output_name}' not found or empty.")

        if len(hash_codes_128_bit) == 2:
            hamming_dist = calculate_hamming_distance(hash_codes_128_bit[0], hash_codes_128_bit[1])
            print(f"\nHamming Distance between 128-bit hash codes:")
            print(f"  Image 1 vs Image 2: {hamming_dist}")
            print(f"  Similarity (1 - normalized Hamming distance): {1 - (hamming_dist / 128.0):.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
