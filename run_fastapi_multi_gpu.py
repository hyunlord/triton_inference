import subprocess
import os
import torch

# 시스템의 GPU 개수 확인
num_gpus = torch.cuda.device_count()

if num_gpus == 0:
    print("No GPUs found. Running FastAPI on CPU.")
    # CPU 전용으로 실행
    subprocess.run(["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"])
else:
    print(f"Running FastAPI with {num_gpus} GPU workers.")
    processes = []
    base_port = 8000 # 첫 번째 워커의 포트
    
    for i in range(num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i) # 각 워커가 사용할 GPU 지정
        
        # 각 워커마다 다른 포트 사용 (벤치마크 클라이언트가 각 포트로 요청을 보내야 함)
        # 또는, 하나의 포트에서 로드 밸런싱을 하려면 Nginx 같은 프록시 필요
        # 여기서는 간단하게 각 워커가 다른 포트에서 리스닝하도록 설정
        worker_port = base_port + i 
        
        print(f"Starting worker {i} on GPU {i} at port {worker_port}")
        
        # 각 워커가 별도의 FastAPI 인스턴스를 실행
        # --reload 옵션은 개발용이므로 벤치마크 시에는 제거
        cmd = ["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", str(worker_port)]
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
    
    print(f"\nFastAPI workers started on ports {base_port} to {base_port + num_gpus - 1}.")
    print("Press Ctrl+C to stop all workers.")

    try:
        # 모든 프로세스가 종료될 때까지 대기
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all FastAPI workers...")
        for p in processes:
            p.terminate() # 프로세스 종료 신호
        for p in processes:
            p.wait() # 프로세스가 종료될 때까지 대기
        print("All FastAPI workers stopped.")