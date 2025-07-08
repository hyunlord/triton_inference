import torch
import numpy as np
import sys

# config.py에서 가져온 이미지 크기
IMAGE_SIZE = 384
BIT_LIST = [8, 16, 32, 48, 64, 128] # 모델 출력 확인용

def test_model_pt(model_path, batch_size, image_size):
    try:
        # 모델 로드 (CPU로 강제)
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        print(f"Model '{model_path}' loaded successfully on CPU.")

        # 더미 입력 생성
        dummy_input = torch.randn(batch_size, 3, image_size, image_size, device='cpu')
        print(f"Dummy input shape: {dummy_input.shape}")

        # 추론 실행
        print("Running inference...")
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print("Inference completed successfully!")
        
        # 출력 형태 확인
        if isinstance(outputs, tuple):
            print(f"Model returned a tuple of {len(outputs)} tensors.")
            for i, output_tensor in enumerate(outputs):
                expected_bit = BIT_LIST[i] if i < len(BIT_LIST) else "unknown"
                print(f"  Output {i} (expected {expected_bit}-bit) shape: {output_tensor.shape}")
                # 간단한 값 확인 (선택 사항)
                # print(f"  Output {i} first 5 values: {output_tensor.flatten()[:5].numpy()}")
        else:
            print(f"Model returned a single tensor with shape: {outputs.shape}")

    except Exception as e:
        print(f"An error occurred during model test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    model_pt_path = "/hanmail/users/rexxa.som/github/triton_inference/triton_models/deep_hashing/1/model.pt" # model.pt 경로
    test_batch_size = 16 # 테스트할 배치 크기

    print(f"Testing model: {model_pt_path} with batch_size={test_batch_size}")
    test_model_pt(model_pt_path, test_batch_size, IMAGE_SIZE)
