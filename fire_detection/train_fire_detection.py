"""
불꽃 감지 YOLO 모델 훈련 스크립트
Fire Detection YOLO Model Training Script
"""

import torch
from ultralytics import YOLO
import os
from pathlib import Path
import time

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
    model_name='yolov8n.pt',  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    epochs=100,
    imgsz=640,
    batch_size=16,
    project_name='fire_detection_runs',
    experiment_name='fire_model'
):
    """
    불꽃 감지 모델 훈련
    
    Args:
        data_yaml (str): 데이터셋 설정 YAML 파일 경로
        model_name (str): 사전 훈련된 YOLO 모델 이름
        epochs (int): 훈련 에포크 수
        imgsz (int): 입력 이미지 크기
        batch_size (int): 배치 크기
        project_name (str): 프로젝트 이름 (결과 저장 폴더)
        experiment_name (str): 실험 이름
    """
    
    print("🔥 불꽃 감지 YOLO 모델 훈련 시작")
    print("=" * 60)
    
    # GPU 상태 확인
    gpu_available = check_gpu_status()
    
    # 데이터셋 YAML 파일 확인
    if not os.path.exists(data_yaml):
        print(f"❌ 오류: 데이터셋 설정 파일을 찾을 수 없습니다: {data_yaml}")
        return None
    
    print(f"\n📊 훈련 설정:")
    print(f"   - 모델: {model_name}")
    print(f"   - 데이터셋: {data_yaml}")
    print(f"   - 에포크: {epochs}")
    print(f"   - 이미지 크기: {imgsz}x{imgsz}")
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 디바이스: {'GPU (CUDA)' if gpu_available else 'CPU'}")
    
    # YOLO 모델 로드
    print(f"\n🤖 모델 로딩 중: {model_name}")
    model = YOLO(model_name)
    
    # 훈련 시작
    print(f"\n🚀 훈련 시작...")
    start_time = time.time()
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=project_name,
            name=experiment_name,
            device='0' if gpu_available else 'cpu',
            
            # 성능 최적화 옵션
            workers=8,  # 데이터 로딩 워커 수
            cache=True,  # 이미지 캐싱 (RAM에 여유가 있을 경우)
            
            # 데이터 증강 옵션
            hsv_h=0.015,  # 색조 변화
            hsv_s=0.7,    # 채도 변화
            hsv_v=0.4,    # 명도 변화
            degrees=0.0,  # 회전
            translate=0.1,  # 이동
            scale=0.5,    # 스케일
            shear=0.0,    # 전단
            perspective=0.0,  # 원근
            flipud=0.0,   # 상하 반전
            fliplr=0.5,   # 좌우 반전
            mosaic=1.0,   # 모자이크 증강
            mixup=0.0,    # 믹스업 증강
            
            # Early stopping
            patience=50,  # 50 에포크 동안 개선이 없으면 중단
            
            # 저장 옵션
            save=True,
            save_period=10,  # 10 에포크마다 저장
            
            # 검증 옵션
            val=True,
            plots=True,  # 결과 플롯 생성
            
            # 추가 옵션
            verbose=True,
            seed=42,  # 재현성을 위한 시드
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 훈련 완료!")
        print(f"⏱️  총 훈련 시간: {training_time/60:.1f}분")
        
        # 결과 경로 출력
        save_dir = Path(project_name) / experiment_name
        best_model = save_dir / 'weights' / 'best.pt'
        last_model = save_dir / 'weights' / 'last.pt'
        
        print(f"\n📁 훈련 결과:")
        print(f"   - 최고 모델: {best_model}")
        print(f"   - 최종 모델: {last_model}")
        print(f"   - 결과 폴더: {save_dir}")
        
        # 모델 검증
        print(f"\n📊 모델 검증 중...")
        metrics = model.val()
        
        print(f"\n📈 성능 메트릭:")
        print(f"   - mAP50: {metrics.box.map50:.4f}")
        print(f"   - mAP50-95: {metrics.box.map:.4f}")
        print(f"   - Precision: {metrics.box.mp:.4f}")
        print(f"   - Recall: {metrics.box.mr:.4f}")
        
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
    
    parser = argparse.ArgumentParser(description='불꽃 감지 YOLO 모델 훈련')
    parser.add_argument('--data', type=str, default='fire_dataset.yaml',
                       help='데이터셋 설정 YAML 파일')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='기본 모델 (yolov8n.pt, yolov8s.pt, yolov8m.pt)')
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
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        experiment_name=args.name
    )
    
    if best_model_path:
        print(f"\n🎉 훈련 완료! 최고 모델: {best_model_path}")
        print(f"\n💡 다음 단계: 동영상 처리에 이 모델을 사용하세요:")
        print(f"   python process_fire_videos.py --model {best_model_path}")

if __name__ == "__main__":
    main()
