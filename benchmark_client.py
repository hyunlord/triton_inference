import tritonclient.http as httpclient
import numpy as np
import time
import argparse
from collections import deque
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import ujson as json
from base64 import b64encode, b64decode

# Triton 서버 정보 (FastAPI도 이 포맷으로 통신 가능)
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "deep_hashing"
MODEL_VERSION = "1"

# config.py에서 가져온 이미지 크기 및 비트 리스트
IMAGE_SIZE = 384
BIT_LIST = [8, 16, 32, 48, 64, 128]

def send_inference_request_triton(server_url, model_name, model_version, batch_size, image_size, bit_list):
    """Triton Inference Server에 단일 추론 요청을 보내고 지연 시간을 반환합니다."""
    # 각 스레드에서 클라이언트 객체를 생성하여 greenlet 스레드 충돌을 방지합니다.
    try:
        with httpclient.InferenceServerClient(url=server_url, verbose=False) as triton_client:
            input_image_data = np.random.rand(batch_size, 3, image_size, image_size).astype(np.float32)

            inputs = []
            inputs.append(httpclient.InferInput("images", input_image_data.shape, "FP32"))
            inputs[0].set_data_from_numpy(input_image_data, binary_data=True)

            outputs = []
            # Request all outputs defined in the model config
            for bit in bit_list:
                outputs.append(httpclient.InferRequestedOutput(f"output_{bit}_bit", binary_data=True))

            request_start_time = time.time()
            success = False
            try:
                response = triton_client.infer(
                    model_name=model_name,
                    inputs=inputs,
                    outputs=outputs,
                    model_version=model_version
                )
                # Check the last output as a proxy for success
                _ = response.as_numpy(f"output_{bit_list[-1]}_bit")
                success = True
            except Exception as e:
                # 에러 메시지를 더 명확하게 출력합니다.
                print(f"An error occurred during inference: {e}")
                success = False
            
            request_end_time = time.time()
            latency_ms = (request_end_time - request_start_time) * 1000
            return latency_ms, batch_size, success

    except Exception as e:
        print(f"Failed to create or use Triton client in thread: {e}")
        return 0, batch_size, False


def send_inference_request_fastapi(server_url_for_worker, model_name, model_version, batch_size, image_size, bit_list_last_element):
    """FastAPI 서버에 단일 추론 요청을 보내고 지연 시간을 반환합니다."""
    input_image_data = np.random.rand(batch_size, 3, image_size, image_size).astype(np.float32)

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
    url = f"{server_url_for_worker}/v2/models/{model_name}/versions/{model_version}/infer"

    request_start_time = time.time()
    success = False
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10.0) # 타임아웃 추가
        if response.status_code == 200:
            response_json = response.json()
            output_data_json = response_json["outputs"][0]["data"][0]
            decoded_output = b64decode(output_data_json)
            _ = np.frombuffer(decoded_output, dtype=np.float32).reshape(response_json["outputs"][0]["shape"])
            success = True
        else:
            success = False
    except Exception as e:
        print(f"An error occurred during benchmark: {e}")
        success = False
    request_end_time = time.time()

    latency_ms = (request_end_time - request_start_time) * 1000
    return latency_ms, batch_size, success


def run_benchmark(server_url, model_name, model_version, num_requests, batch_size, image_size, num_concurrent_clients, server_type, fastapi_worker_ports):
    latencies = deque() # 밀리초 단위
    total_images_processed = 0
    successful_requests = 0
    failed_requests = 0
    
    try:
        if server_type == "triton":
            parsed_url = server_url.replace("http://", "").replace("https://", "")
            # 메인 스레드에서 임시 클라이언트를 생성하여 서버 상태를 확인합니다.
            try:
                with httpclient.InferenceServerClient(url=parsed_url, verbose=False) as client:
                    print(f"Checking server readiness: {client.is_server_ready()}")
                    print(f"Checking model readiness for {model_name}: {client.is_model_ready(model_name)}")
            except Exception as client_init_e:
                print(f"Error initializing Triton client for health check: {client_init_e}")
                raise
            
            send_request_func = send_inference_request_triton
        elif server_type == "fastapi":
            send_request_func = send_inference_request_fastapi
            
            # FastAPI 워커 URL 리스트 생성
            fastapi_urls = []
            if fastapi_worker_ports:
                base_host = server_url.split("://")[0] + "://" + server_url.split("://")[1].split(":")[0] # http://localhost
                for port in fastapi_worker_ports.split(','):
                    fastapi_urls.append(f"{base_host}:{port.strip()}")
            else:
                fastapi_urls.append(server_url)
            
            if not fastapi_urls:
                raise ValueError("No FastAPI worker ports specified or derived.")
            print(f"FastAPI worker URLs: {fastapi_urls}")

            # 헬스 체크는 첫 번째 URL로만 시도
            try:
                health_resp = requests.get(f"{fastapi_urls[0]}/v2/health/ready")
                model_health_resp = requests.get(f"{fastapi_urls[0]}/v2/models/{model_name}/ready")
                print(f"Checking server readiness (FastAPI): {health_resp.status_code == 200}")
                print(f"Checking model readiness for {model_name} (FastAPI): {model_health_resp.status_code == 200}")
            except Exception as e:
                print(f"FastAPI health check failed: {e}")
                raise
        else:
            raise ValueError("Invalid server_type. Must be 'triton' or 'fastapi'.")

        print(f"\n--- Starting Benchmark ---")
        print(f"Server Type: {server_type.upper()}")
        print(f"Server URL(s): {server_url if server_type == 'triton' else fastapi_urls}") # 출력 변경
        print(f"Model: {model_name}, Version: {model_version}")
        print(f"Number of requests: {num_requests}")
        print(f"Batch size per request: {batch_size}")
        print(f"Image size: {image_size}x{image_size}")
        print(f"Number of concurrent clients: {num_concurrent_clients}")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_concurrent_clients) as executor:
            futures = []
            for i in range(num_requests):
                if server_type == "triton":
                    # 각 작업에 client_instance 대신 parsed_url을 전달합니다.
                    futures.append(executor.submit(send_request_func, parsed_url, model_name, model_version, batch_size, image_size, BIT_LIST))
                elif server_type == "fastapi":
                    worker_url = fastapi_urls[i % len(fastapi_urls)]
                    futures.append(executor.submit(send_request_func, worker_url, model_name, model_version, batch_size, image_size, BIT_LIST[-1]))

            for future in tqdm(as_completed(futures), total=num_requests, desc="Benchmarking"):
                latency_ms, processed_batch_size, success = future.result()
                if success:
                    latencies.append(latency_ms)
                    total_images_processed += processed_batch_size
                    successful_requests += 1
                else:
                    failed_requests += 1

        end_time = time.time()
        total_duration_sec = end_time - start_time

        if successful_requests > 0:
            all_latencies = np.array(latencies)
            avg_latency_ms = np.mean(all_latencies)
            std_latency_ms = np.std(all_latencies)
            p90_latency_ms = np.percentile(all_latencies, 90)
            p95_latency_ms = np.percentile(all_latencies, 95)
            p99_latency_ms = np.percentile(all_latencies, 99)
        else:
            avg_latency_ms = p90_latency_ms = p95_latency_ms = p99_latency_ms = float('nan')
            std_latency_ms = float('nan')

        rps = num_requests / total_duration_sec
        ips = total_images_processed / total_duration_sec
        error_rate = (failed_requests / num_requests) * 100 if num_requests > 0 else 0

        print(f"\n--- Benchmark Results ---")
        print(f"Total duration: {total_duration_sec:.2f} seconds")
        print(f"Total requests sent: {num_requests}")
        print(f"Successful requests: {successful_requests}")
        print(f"Failed requests: {failed_requests}")
        print(f"Error Rate: {error_rate:.2f}%")
        print(f"Total images processed: {total_images_processed}")
        print(f"Average Latency: {avg_latency_ms:.2f} ms/request")
        print(f"Standard Deviation of Latency: {std_latency_ms:.2f} ms")
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
    parser.add_argument("--fastapi_worker_ports", type=str, default="",
                        help="Comma-separated list of ports for FastAPI workers (e.g., '8000,8001,8002,8003'). Only used with --server_type fastapi.")
    args = parser.parse_args()

    run_benchmark(args.server_url, args.model_name, args.model_version,
                  args.num_requests, args.batch_size, args.image_size, args.num_concurrent_clients, args.server_type, args.fastapi_worker_ports)
