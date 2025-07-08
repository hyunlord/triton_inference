import tritonclient.http as httpclient
import numpy as np
import time
import argparse
from collections import deque
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests # 추가
import json
from base64 import b64encode, b64decode # 추가

# Triton 서버 정보 (FastAPI도 이 포맷으로 통신 가능)
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "deep_hashing"
MODEL_VERSION = "1"

# config.py에서 가져온 이미지 크기 및 비트 리스트
IMAGE_SIZE = 384
BIT_LIST = [8, 16, 32, 48, 64, 128]

def send_inference_request_triton(triton_client, model_name, model_version, batch_size, image_size, bit_list_last_element):
    """Triton Inference Server에 단일 추론 요청을 보내고 지연 시간을 반환합니다."""
    input_image_data = np.random.rand(batch_size, 3, image_size, image_size).astype(np.float32)

    inputs = []
    inputs.append(httpclient.InferInput("images", input_image_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_image_data, binary_data=True)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput(f"output_{bit_list_last_element}_bit", binary_data=True))

    request_start_time = time.time()
    response = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        model_version=model_version
    )
    request_end_time = time.time()

    latency_ms = (request_end_time - request_start_time) * 1000
    # 응답 데이터 확인 (선택 사항, 성능에 영향)
    _ = response.as_numpy(f"output_{bit_list_last_element}_bit") # 응답 파싱 포함
    return latency_ms, batch_size

def send_inference_request_fastapi(server_url, model_name, model_version, batch_size, image_size, bit_list_last_element):
    """FastAPI 서버에 단일 추론 요청을 보내고 지연 시간을 반환합니다."""
    input_image_data = np.random.rand(batch_size, 3, image_size, image_size).astype(np.float32)

    # NumPy 배열을 Base64로 인코딩하여 JSON 페이로드 생성
    encoded_input = b64encode(input_image_data.tobytes()).decode('utf-8')
    
    payload = {
        "inputs": [
            {
                "name": "images",
                "shape": list(input_image_data.shape),
                "datatype": "FP32",
                "data": [encoded_input]
            }
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    url = f"{server_url}/v2/models/{model_name}/versions/{model_version}/infer"

    request_start_time = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    request_end_time = time.time()

    latency_ms = (request_end_time - request_start_time) * 1000

    if response.status_code != 200:
        raise Exception(f"FastAPI server returned error: {response.status_code} - {response.text}")
    
    # 응답 데이터 확인 (선택 사항, 성능에 영향)
    response_json = response.json()
    
    output_data_json = response_json["outputs"][0]["data"][0]
    decoded_output = b64decode(output_data_json)
    _ = np.frombuffer(decoded_output, dtype=np.float32).reshape(response_json["outputs"][0]["shape"])

    return latency_ms, batch_size

def run_benchmark(server_url, model_name, model_version, num_requests, batch_size, image_size, num_concurrent_clients, server_type):
    latencies = deque() # 밀리초 단위
    total_images_processed = 0

    try:
        if server_type == "triton":
            # URL에서 스킴 제거
            parsed_url = server_url.replace("http://", "").replace("https://", "")
            client_instance = httpclient.InferenceServerClient(url=parsed_url)
            send_request_func = send_inference_request_triton
            # Triton client의 헬스 체크 사용
            print(f"Checking server readiness: {client_instance.is_server_ready()}")
            print(f"Checking model readiness for {model_name}: {client_instance.is_model_ready(model_name)}")
        elif server_type == "fastapi":
            # FastAPI는 requests 라이브러리를 직접 사용하므로 별도의 클라이언트 인스턴스 불필요
            client_instance = None # 더미
            send_request_func = send_inference_request_fastapi
            # FastAPI의 헬스 체크는 requests로 직접 호출
            try:
                health_resp = requests.get(f"{server_url}/v2/health/ready")
                model_health_resp = requests.get(f"{server_url}/v2/models/{model_name}/ready")
                print(f"Checking server readiness (FastAPI): {health_resp.status_code == 200}")
                print(f"Checking model readiness for {model_name} (FastAPI): {model_health_resp.status_code == 200}")
            except Exception as e:
                print(f"FastAPI health check failed: {e}")
                raise
        else:
            raise ValueError("Invalid server_type. Must be 'triton' or 'fastapi'.")

        print(f"\n--- Starting Benchmark ---")
        print(f"Server Type: {server_type.upper()}")
        print(f"Server URL: {server_url}")
        print(f"Model: {model_name}, Version: {model_version}")
        print(f"Number of requests: {num_requests}")
        print(f"Batch size per request: {batch_size}")
        print(f"Image size: {image_size}x{image_size}")
        print(f"Number of concurrent clients: {num_concurrent_clients}")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_concurrent_clients) as executor:
            futures = []
            for _ in range(num_requests):
                if server_type == "triton":
                    futures.append(executor.submit(send_request_func, client_instance, model_name, model_version, batch_size, image_size, BIT_LIST[-1]))
                elif server_type == "fastapi":
                    futures.append(executor.submit(send_request_func, server_url, model_name, model_version, batch_size, image_size, BIT_LIST[-1]))

            for future in tqdm(as_completed(futures), total=num_requests, desc="Benchmarking"):
                latency_ms, processed_batch_size = future.result()
                latencies.append(latency_ms)
                total_images_processed += processed_batch_size

        end_time = time.time()
        total_duration_sec = end_time - start_time

        all_latencies = np.array(latencies)
        avg_latency_ms = np.mean(all_latencies)
        p90_latency_ms = np.percentile(all_latencies, 90)
        p95_latency_ms = np.percentile(all_latencies, 95)
        p99_latency_ms = np.percentile(all_latencies, 99)

        rps = num_requests / total_duration_sec
        ips = total_images_processed / total_duration_sec

        print(f"\n--- Benchmark Results ---")
        print(f"Total duration: {total_duration_sec:.2f} seconds")
        print(f"Total requests: {num_requests}")
        print(f"Total images processed: {total_images_processed}")
        print(f"Average Latency: {avg_latency_ms:.2f} ms/request")
        print(f"P90 Latency: {p90_latency_ms:.2f} ms/request")
        print(f"P95 Latency: {p95_latency_ms:.2f} ms/request")
        print(f"P99 Latency: {p99_latency_ms:.2f} ms/request")
        print(f"Requests Per Second (RPS): {rps:.2f}")
        print(f"Images Per Second (IPS): {ips:.2f}")

    except Exception as e:
        print(f"An error occurred during benchmark: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark client for Triton/FastAPI inference servers.")
    parser.add_argument("--server_url", type=str, default=TRITON_SERVER_URL,
                        help="URL of the inference server (e.g., http://localhost:8000)")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME,
                        help="Name of the model to benchmark")
    parser.add_argument("--model_version", type=str, default=MODEL_VERSION,
                        help="Version of the model to benchmark")
    parser.add_argument("--num_requests", type=int, default=100,
                        help="Number of inference requests to send")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per inference request")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE,
                        help="Image size (height and width) for input")
    parser.add_argument("--num_concurrent_clients", type=int, default=1,
                        help="Number of concurrent clients (threads) to simulate concurrent requests")
    parser.add_argument("--server_type", type=str, choices=["triton", "fastapi"], required=True,
                        help="Type of server to benchmark: 'triton' or 'fastapi'")
    args = parser.parse_args()

    run_benchmark(args.server_url, args.model_name, args.model_version,
                  args.num_requests, args.batch_size, args.image_size, args.num_concurrent_clients, args.server_type)