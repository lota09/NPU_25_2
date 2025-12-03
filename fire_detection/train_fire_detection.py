"""
불꽃 감지 YOLOv7 모델 훈련 스크립트
Fire Detection YOLOv7 Model Training Script
"""

import torch
import os
from pathlib import Path
import time
import subprocess
import sys

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

def train_fire_detection_model(
    data_yaml='fire_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch_size=16,
    weights='yolov7.pt',
    cfg='cfg/training/yolov7.yaml',
    experiment_name='fire_model'
):
    """
    불꽃 감지 YOLOv7 모델 훈련
    
    Note: YOLOv7은 자동으로 best.pt (최고 mAP 모델)를 저장합니다.
          과적합 방지는 best.pt 사용으로 자동 처리됩니다.
    """
    
    print("🔥 불꽃 감지 YOLOv7 모델 훈련 시작")
    print("=" * 60)
    
    # GPU 상태 확인
    gpu_available = check_gpu_status()
    
    # YOLOv7 디렉토리
    yolov7_dir = Path('yolov7')
    
    # 데이터셋 YAML 파일 확인
    if not os.path.exists(data_yaml):
        print(f"❌ 오류: 데이터셋 설정 파일을 찾을 수 없습니다: {data_yaml}")
        return None
    
    print(f"\n📊 훈련 설정:")
    print(f"   - 모델: YOLOv7")
    print(f"   - 데이터셋: {data_yaml}")
    print(f"   - 에포크: {epochs}")
    print(f"   - 이미지 크기: {imgsz}x{imgsz}")
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 과적합 방지: best.pt 자동 저장 (최고 mAP 모델)")
    print(f"   - 디바이스: {'GPU (CUDA)' if gpu_available else 'CPU'}")
    
    # 훈련 시작
    print(f"\n🚀 훈련 시작...")
    start_time = time.time()
    
    try:
        # 절대 경로로 변환
        data_yaml_abs = str(Path(data_yaml).absolute())
        
        # YOLOv7 train.py 실행
        cmd = [
            sys.executable,
            str(yolov7_dir / 'train.py'),
            '--workers', '8',
            '--device', '0',  # GPU 강제 사용
            '--batch-size', str(batch_size),
            '--epochs', str(epochs),
            '--data', data_yaml_abs,
            '--img', str(imgsz),
            '--cfg', str(yolov7_dir / cfg),
            '--weights', str(yolov7_dir / weights),
            '--name', experiment_name,
            '--hyp', str(yolov7_dir / 'data/hyp.scratch.p5.yaml'),
            '--project', str(Path.cwd() / 'runs/train')
        ]
        
        print(f"실행 명령: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 훈련 완료!")
        print(f"⏱️  총 훈련 시간: {training_time/60:.1f}분")
        
        # 결과 경로
        save_dir = yolov7_dir / 'runs' / 'train' / experiment_name
        best_model = save_dir / 'weights' / 'best.pt'
        
        print(f"\n📁 훈련 결과:")
        print(f"   - 최고 모델: {best_model}")
        
        return str(best_model)
        
    except KeyboardInterrupt:
        print("\n⏹️  훈련이 사용자에 의해 중단되었습니다.")
        return None
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='불꽃 감지 YOLOv7 모델 훈련')
    parser.add_argument('--data', type=str, default='fire_data.yaml',
                       help='데이터셋 설정 YAML 파일')
    parser.add_argument('--weights', type=str, default='yolov7.pt',
                       help='YOLOv7 사전 훈련 가중치')
    parser.add_argument('--epochs', type=int, default=100,
                       help='훈련 에포크 수')
    parser.add_argument('--batch', type=int, default=16,
                       help='배치 크기 (GPU 메모리에 따라 조정)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='이미지 크기')
    parser.add_argument('--name', type=str, default='fire_model',
                       help='실험 이름')
    
    args = parser.parse_args()
    
    # 훈련 시작
    best_model_path = train_fire_detection_model(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        weights=args.weights,
        experiment_name=args.name
    )
    
    if best_model_path:
        print(f"\n🎉 훈련 완료! 최고 모델: {best_model_path}")
        print(f"\n💡 다음 단계: 동영상 처리에 이 모델을 사용하세요:")
        print(f"   python process_fire_videos.py --model {best_model_path}")

if __name__ == "__main__":
    main()
