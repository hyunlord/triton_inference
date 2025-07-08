import subprocess
import os
import torch
import sys
import shutil # shutil 모듈 임포트

# 시스템의 GPU 개수 확인
num_gpus = torch.cuda.device_count()

# 현재 Python 환경의 uvicorn 실행 파일 경로를 찾습니다.
uvicorn_executable = shutil.which('uvicorn')

if uvicorn_executable is None:
    print("Error: 'uvicorn' executable not found in PATH.")
    print("Please ensure uvicorn is installed in your active environment (e.g., 'pip install uvicorn')")
    print("and that your shell's PATH is correctly configured for your virtual environment.")
    sys.exit(1) # 스크립트 종료

if num_gpus == 0:
    print("No GPUs found. Running FastAPI on CPU.")
    subprocess.run([uvicorn_executable, "fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"])
else:
    print(f"Running FastAPI with {num_gpus} GPU workers.")
    processes = []
    base_port = 8000 # 첫 번째 워커의 포트
    
    for i in range(num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i) # 각 워커가 사용할 GPU 지정
        
        worker_port = base_port + i 
        
        print(f"Starting worker {i} on GPU {i} at port {worker_port}")
        
        cmd = [uvicorn_executable, "fastapi_server:app", "--host", "0.0.0.0", "--port", str(worker_port)]
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
    
    print(f"\nFastAPI workers started on ports {base_port} to {base_port + num_gpus - 1}.")
    print("Press Ctrl+C to stop all workers.")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all FastAPI workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        print("All FastAPI workers stopped.")