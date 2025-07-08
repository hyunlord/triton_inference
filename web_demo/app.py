import os
from flask import Flask, request, render_template, redirect, url_for
import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Triton 서버 정보
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "deep_hashing"
MODEL_VERSION = "1"
IMAGE_SIZE = 384
BIT_LIST = [8, 16, 32, 48, 64, 128]

def preprocess_image(image_file, image_size):
    """이미지 파일을 로드하고 Triton 서버 입력 형식에 맞게 전처리합니다."""
    try:
        img = Image.open(image_file).convert("RGB")
        img = img.resize((image_size, image_size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32)
        img_np = img_np.transpose((2, 0, 1)) # HWC to CHW
        img_np = img_np / 255.0 # Normalize to [0, 1]
        img_np = np.expand_dims(img_np, axis=0) # Add batch dimension
        return img_np
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def calculate_hamming_distance(hash1, hash2):
    """두 이진 해시 코드 간의 해밍 거리를 계산합니다."""
    return np.sum(hash1 != hash2)

def hash_to_binary_string(hash_array):
    """해시 배열을 이진 문자열로 변환합니다 (-1은 0으로)."""
    return ''.join(['1' if x == 1 else '0' for x in hash_array])

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image1' not in request.files or 'image2' not in request.files:
        return redirect(request.url)

    image1_file = request.files['image1']
    image2_file = request.files['image2']

    if image1_file.filename == '' or image2_file.filename == '':
        return redirect(request.url)

    if image1_file and image2_file:
        try:
            # 이미지 전처리
            image1_data = preprocess_image(image1_file, IMAGE_SIZE)
            image2_data = preprocess_image(image2_file, IMAGE_SIZE)

            if image1_data is None or image2_data is None:
                return "Error: Could not process images.", 400

            # 두 이미지를 하나의 배치로 결합
            batch_image_data = np.concatenate((image1_data, image2_data), axis=0)

            # Triton HTTP 클라이언트 생성
            triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

            # 입력 객체 생성
            inputs = []
            inputs.append(httpclient.InferInput("images", batch_image_data.shape, "FP32"))
            inputs[0].set_data_from_numpy(batch_image_data, binary_data=True)

            # 출력 객체 생성
            outputs = []
            for bit in BIT_LIST:
                outputs.append(httpclient.InferRequestedOutput(f"output_{bit}_bit", binary_data=True))

            # 추론 요청 보내기
            response = triton_client.infer(
                model_name=MODEL_NAME,
                inputs=inputs,
                outputs=outputs,
                model_version=MODEL_VERSION
            )

            # 결과 확인 및 해시 코드 비교
            results = {}
            hash_codes_128_bit_raw = [] # 해밍 거리 계산을 위한 원본 해시 배열 저장

            for bit in BIT_LIST:
                output_name = f"output_{bit}_bit"
                output_data = response.as_numpy(output_name)

                if output_data is not None:
                    output_image1 = output_data[0]
                    output_image2 = output_data[1]

                    signed_hash_image1 = np.sign(output_image1)
                    signed_hash_image2 = np.sign(output_image2)

                    # 각 비트 길이별 해밍 거리 계산 및 저장
                    hamming_dist_current_bit = calculate_hamming_distance(signed_hash_image1, signed_hash_image2)
                    results[f'hamming_distance_{bit}_bit'] = int(hamming_dist_current_bit)

                    # 이진 문자열로 변환하여 저장
                    results[f'{bit}_bit_hash_image1_str'] = hash_to_binary_string(signed_hash_image1)
                    results[f'{bit}_bit_hash_image2_str'] = hash_to_binary_string(signed_hash_image2)

                    # 비교를 위한 HTML 문자열 생성
                    compared_html = ""
                    for i in range(bit):
                        char1 = '1' if signed_hash_image1[i] == 1 else '0'
                        char2 = '1' if signed_hash_image2[i] == 1 else '0'
                        if char1 != char2:
                            compared_html += f"<span style='color:red;'>{char1}</span>"
                        else:
                            compared_html += char1
                    results[f'{bit}_bit_compared_html'] = compared_html

                    if bit == 128:
                        hash_codes_128_bit_raw.append(signed_hash_image1)
                        hash_codes_128_bit_raw.append(signed_hash_image2)
                else:
                    results[f'{bit}_bit_hash_image1_str'] = "N/A"
                    results[f'{bit}_bit_hash_image2_str'] = "N/A"
                    results[f'{bit}_bit_compared_html'] = "N/A"

            if len(hash_codes_128_bit_raw) == 2:
                hamming_dist = calculate_hamming_distance(hash_codes_128_bit_raw[0], hash_codes_128_bit_raw[1])
                results['hamming_distance_128_bit'] = int(hamming_dist)
                results['similarity_128_bit'] = float(1 - (hamming_dist / 128.0))
            else:
                results['hamming_distance_128_bit'] = "N/A"
                results['similarity_128_bit'] = "N/A"

            return render_template('results.html', results=results)

        except Exception as e:
            return f"An error occurred during inference: {e}", 500

    return "Invalid request", 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)