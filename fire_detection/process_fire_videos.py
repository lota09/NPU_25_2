"""
훈련된 불꽃 감지 모델로 동영상 처리
Process Videos with Trained Fire Detection Model
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np
import argparse
import os
from pathlib import Path
import time
from typing import List

def check_gpu_status():
    """GPU 상태 확인 및 정보 출력"""
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        compute_cap = torch.cuda.get_device_capability(0)
        
        print(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"⚡ Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        
        if compute_cap[0] >= 7:
            print("🚀 Tensor Core 지원")
        
        return True
    else:
        print("🖥️  GPU: 사용 불가 (CPU 모드)")
        return False

def optimize_model_for_gpu(model, use_gpu=True):
    """GPU 최적화 적용"""
    device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
    
    if device == 'cuda':
        print("🚀 GPU 최적화 적용 중...")
        model.to(device)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("   ✅ GPU 최적화 완료")
        
        # GPU 워밍업
        print("🔥 GPU 워밍업 중...")
        try:
            dummy_input = torch.zeros((640, 640, 3), dtype=torch.uint8, device='cpu')
            with torch.no_grad():
                _ = model(dummy_input, verbose=False)
            torch.cuda.synchronize()
            print("   ✅ GPU 워밍업 완료")
        except Exception as e:
            print(f"   ⚠️  워밍업 건너뜀: {str(e)}")
    else:
        print("🔧 CPU 모드")
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    
    return device

def process_single_video(
    input_video_path: str,
    output_video_path: str,
    model,
    device: str,
    confidence: float = 0.5
):
    """
    단일 동영상 처리
    
    Args:
        input_video_path: 입력 동영상 경로
        output_video_path: 출력 동영상 경로
        model: YOLO 모델
        device: 디바이스 ('cuda' 또는 'cpu')
        confidence: 신뢰도 임계값
    """
    
    print(f"\n📹 처리 중: {Path(input_video_path).name}")
    print("-" * 60)
    
    # 비디오 캡처
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ 오류: 동영상을 열 수 없습니다: {input_video_path}")
        return False
    
    # 비디오 속성
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📺 동영상 정보:")
    print(f"   - 해상도: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - 총 프레임: {total_frames}")
    
    # 비디오 라이터
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    fire_detected_frames = 0
    processing_times = []
    inference_times = []
    total_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_start_time = time.time()
            
            # 추론
            inference_start = time.time()
            
            if device == 'cuda':
                results = model(frame, conf=confidence, verbose=False, device=device)
                torch.cuda.synchronize()
            else:
                results = model(frame, conf=confidence, verbose=False, device=device)
            
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            # 불꽃 감지 여부 확인
            detections = results[0].boxes
            if len(detections) > 0:
                fire_detected_frames += 1
            
            # 결과를 프레임에 그리기
            annotated_frame = results[0].plot()
            
            # 불꽃 감지 시 경고 텍스트 추가
            if len(detections) > 0:
                cv2.putText(
                    annotated_frame,
                    "🔥 FIRE DETECTED!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
            
            # 처리된 프레임 저장
            out.write(annotated_frame)
            
            frame_time = time.time() - frame_start_time
            processing_times.append(frame_time)
            
            # 진행률 표시
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                avg_inference = np.mean(inference_times[-30:]) * 1000
                avg_fps = 1 / np.mean(processing_times[-30:]) if processing_times else 0
                
                print(f"📊 진행: {progress:.1f}% | ⚡ {avg_inference:.1f}ms | 🎬 {avg_fps:.1f} FPS | 🔥 감지: {fire_detected_frames}/{frame_count}", end="")
                
                if device == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f" | 💾 {gpu_memory:.1f}GB")
                else:
                    print()
    
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
        return False
    
    finally:
        cap.release()
        out.release()
        
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # 성능 리포트
        total_time = time.time() - total_start_time
        if frame_count > 0:
            avg_fps = frame_count / total_time
            avg_inference = np.mean(inference_times) * 1000
            fire_percentage = (fire_detected_frames / frame_count) * 100
            
            print(f"\n📈 처리 완료:")
            print(f"   ⏱️  처리 시간: {total_time:.2f}초")
            print(f"   🎯 평균 FPS: {avg_fps:.2f}")
            print(f"   ⚡ 평균 추론: {avg_inference:.1f}ms")
            print(f"   🔥 불꽃 감지율: {fire_percentage:.1f}% ({fire_detected_frames}/{frame_count} 프레임)")
            print(f"   📁 저장 위치: {output_video_path}")
    
    return True

def process_multiple_videos(
    video_paths: List[str],
    model_path: str,
    output_dir: str = "fire_detected_videos",
    confidence: float = 0.5,
    use_gpu: bool = True
):
    """
    여러 동영상을 배치 처리
    
    Args:
        video_paths: 처리할 동영상 경로 리스트
        model_path: 훈련된 모델 경로
        output_dir: 출력 디렉토리
        confidence: 신뢰도 임계값
        use_gpu: GPU 사용 여부
    """
    
    print("🔥 불꽃 감지 동영상 처리 시작")
    print("=" * 60)
    
    # GPU 상태 확인
    gpu_available = check_gpu_status()
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"📁 출력 디렉토리: {output_path}")
    
    # 모델 로드
    print(f"\n🤖 모델 로딩: {model_path}")
    model_start_time = time.time()
    model = YOLO(model_path)
    
    # GPU 최적화
    device = optimize_model_for_gpu(model, use_gpu and gpu_available)
    
    model_load_time = time.time() - model_start_time
    print(f"✅ 모델 로드 완료 ({model_load_time:.2f}초)")
    
    # 각 동영상 처리
    total_start = time.time()
    success_count = 0
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n{'='*60}")
        print(f"📹 동영상 {i}/{len(video_paths)}")
        
        if not os.path.exists(video_path):
            print(f"❌ 파일이 존재하지 않습니다: {video_path}")
            continue
        
        # 출력 파일명 생성
        video_name = Path(video_path).stem
        output_video = output_path / f"{video_name}_fire_detected.mp4"
        
        # 동영상 처리
        success = process_single_video(
            input_video_path=video_path,
            output_video_path=str(output_video),
            model=model,
            device=device,
            confidence=confidence
        )
        
        if success:
            success_count += 1
    
    # 전체 처리 완료
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"✅ 전체 처리 완료!")
    print(f"📊 처리 결과: {success_count}/{len(video_paths)} 성공")
    print(f"⏱️  총 소요 시간: {total_time/60:.1f}분")
    print(f"📁 결과 위치: {output_path}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='훈련된 불꽃 감지 모델로 동영상 처리')
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='훈련된 YOLO 모델 경로 (예: fire_detection_runs/fire_model/weights/best.pt)')
    parser.add_argument('--videos', '-v', type=str, nargs='+',
                       help='처리할 동영상 파일들 (여러 개 가능)')
    parser.add_argument('--video-dir', type=str,
                       help='동영상이 있는 디렉토리 (모든 .mp4 파일 처리)')
    parser.add_argument('--output-dir', '-o', type=str, default='fire_detected_videos',
                       help='출력 디렉토리 (기본값: fire_detected_videos)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='신뢰도 임계값 (기본값: 0.5)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='GPU 사용 안함')
    
    args = parser.parse_args()
    
    # 모델 파일 확인
    if not os.path.exists(args.model):
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {args.model}")
        return
    
    # 처리할 동영상 목록 생성
    video_paths = []
    
    if args.videos:
        video_paths.extend(args.videos)
    
    if args.video_dir:
        video_dir = Path(args.video_dir)
        if video_dir.exists():
            video_paths.extend([str(f) for f in video_dir.glob('*.mp4')])
            video_paths.extend([str(f) for f in video_dir.glob('*.avi')])
            video_paths.extend([str(f) for f in video_dir.glob('*.mov')])
        else:
            print(f"⚠️  경고: 디렉토리를 찾을 수 없습니다: {args.video_dir}")
    
    if not video_paths:
        print("❌ 오류: 처리할 동영상이 없습니다.")
        print("   --videos 또는 --video-dir 옵션을 사용하세요.")
        return
    
    # 중복 제거
    video_paths = list(set(video_paths))
    
    print(f"📋 처리할 동영상 ({len(video_paths)}개):")
    for vp in video_paths:
        print(f"   - {Path(vp).name}")
    
    # 동영상 처리 실행
    process_multiple_videos(
        video_paths=video_paths,
        model_path=args.model,
        output_dir=args.output_dir,
        confidence=args.confidence,
        use_gpu=not args.no_gpu
    )

if __name__ == "__main__":
    # 인자 없이 실행 시 assets 폴더의 동영상 자동 처리
    import sys
    
    if len(sys.argv) == 1:
        print("🔥 불꽃 감지 동영상 처리 프로그램")
        print("\n📖 사용 예시:")
        print("python process_fire_videos.py --model fire_detection_runs/fire_model/weights/best.pt --video-dir assets")
        print("python process_fire_videos.py --model best.pt --videos video1.mp4 video2.mp4")
        print("python process_fire_videos.py --model best.pt --video-dir assets --confidence 0.7")
        
        # 자동 실행 시도
        assets_dir = Path("assets")
        if assets_dir.exists():
            video_files = list(assets_dir.glob("*.mp4"))
            if video_files:
                print(f"\n📹 assets 폴더에서 {len(video_files)}개의 동영상 발견:")
                for vf in video_files:
                    print(f"   - {vf.name}")
                
                # 최신 훈련 모델 찾기
                runs_dir = Path("fire_detection_runs")
                if runs_dir.exists():
                    best_models = list(runs_dir.glob("*/weights/best.pt"))
                    if best_models:
                        latest_model = max(best_models, key=lambda p: p.stat().st_mtime)
                        print(f"\n🤖 최신 모델 발견: {latest_model}")
                        print(f"\n자동 실행을 원하시면 다음 명령을 사용하세요:")
                        print(f"python process_fire_videos.py --model {latest_model} --video-dir assets")
    else:
        main()
