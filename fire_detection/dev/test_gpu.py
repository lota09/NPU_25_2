import torch

def check_gpu_status():
    """GPU 상태 확인"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"✅ CUDA 버전: {torch.version.cuda}")
        return True
    else:
        print("⚠️  GPU를 사용할 수 없습니다. CPU로 훈련합니다.")
        return False
    
if __name__ == "__main__":
    check_gpu_status()