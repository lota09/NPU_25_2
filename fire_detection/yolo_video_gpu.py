import cv2
import torch
from ultralytics import YOLO
import numpy as np
import argparse
import os
from pathlib import Path
import time

def check_gpu_status():
    """GPU 상태 확인 및 정보 출력"""
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        compute_cap = torch.cuda.get_device_capability(0)
        
        print(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"⚡ Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        
        # Tensor Core 지원 확인
        if compute_cap[0] >= 7:
            print("🚀 Tensor Core 지원 - 혼합 정밀도(FP16) 사용 가능")
        
        return True, gpu_name
    else:
        print("🖥️  GPU: 사용 불가 (CPU 모드)")
        return False, "CPU"

def optimize_model_for_gpu(model, use_gpu=True):
    """GPU 최적화 적용"""
    device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
    
    if device == 'cuda':
        print("🚀 GPU 최적화 적용 중...")
        
        # GPU로 모델 이동
        model.to(device)
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        
        # CUDNN 최적화
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        print("   ✅ GPU 최적화 설정 완료")
        
        # GPU 워밍업 (FP16 없이)
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
        print("🔧 CPU 모드 - 스레드 최적화 적용")
        # CPU 최적화
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    
    return device

def process_video_with_yolo(input_video_path, output_video_path, model_path='yolov10n.pt', confidence=0.5, use_gpu=True):
    """
    YOLOv10을 사용하여 동영상에서 물체인식을 수행하고 결과를 저장합니다.
    
    Args:
        input_video_path (str): 입력 동영상 파일 경로
        output_video_path (str): 출력 동영상 파일 경로
        model_path (str): YOLO 모델 파일 경로 (기본값: 'yolov10n.pt')
        confidence (float): 신뢰도 임계값 (기본값: 0.5)
        use_gpu (bool): GPU 사용 여부 (기본값: True)
    """
    
    print(f"🎯 YOLOv10 동영상 물체인식 (GPU 가속)")
    print("=" * 50)
    
    # GPU 상태 확인
    gpu_available, gpu_info = check_gpu_status()
    
    # YOLO 모델 로드
    print(f"🤖 모델 로딩 중: {model_path}")
    model_start_time = time.time()
    
    model = YOLO(model_path)
    
    # GPU 최적화 적용
    device = optimize_model_for_gpu(model, use_gpu and gpu_available)
    
    model_load_time = time.time() - model_start_time
    print(f"✅ 모델 로드 완료! ({model_load_time:.2f}초)")
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ 오류: 동영상 파일을 열 수 없습니다: {input_video_path}")
        return
    
    # 비디오 속성 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📺 동영상 정보:")
    print(f"   - 해상도: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - 총 프레임 수: {total_frames}")
    print(f"   - 예상 소요시간: {total_frames / fps / (10 if device == 'cuda' else 2):.1f}초")
    
    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
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
            
            # GPU 최적화 추론
            inference_start = time.time()
            
            if device == 'cuda':
                # GPU에서 안전한 추론
                results = model(frame, conf=confidence, verbose=False, device=device)
                torch.cuda.synchronize()  # GPU 동기화
            else:
                results = model(frame, conf=confidence, verbose=False, device=device)
            
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            # 결과를 프레임에 그리기
            annotated_frame = results[0].plot()
            
            # 처리된 프레임을 출력 비디오에 쓰기
            out.write(annotated_frame)
            
            frame_time = time.time() - frame_start_time
            processing_times.append(frame_time)
            
            # 진행률 및 성능 표시
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                avg_inference = np.mean(inference_times[-30:]) * 1000  # ms
                avg_fps = 1 / np.mean(processing_times[-30:]) if processing_times else 0
                
                print(f"📊 진행률: {progress:.1f}% | ⚡ 추론: {avg_inference:.1f}ms | 🎬 FPS: {avg_fps:.1f}", end="")
                
                if device == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    gpu_usage = (gpu_memory / torch.cuda.get_device_properties(0).total_memory * 1024**3) * 100
                    print(f" | 💾 GPU: {gpu_memory:.1f}GB ({gpu_usage:.1f}%)")
                else:
                    print()
    
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
    
    finally:
        # 리소스 해제
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # GPU 메모리 정리
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # 최종 성능 리포트
        total_time = time.time() - total_start_time
        if frame_count > 0:
            avg_fps = frame_count / total_time
            avg_inference = np.mean(inference_times) * 1000
            speedup = avg_fps / fps  # 실시간 대비 속도
            
            print(f"\n📈 최종 성능 리포트:")
            print(f"   ⏱️  총 처리 시간: {total_time:.2f}초")
            print(f"   🎯 평균 FPS: {avg_fps:.2f}")
            print(f"   ⚡ 평균 추론 시간: {avg_inference:.1f}ms")
            print(f"   🚀 실시간 대비 속도: {speedup:.1f}x")
            print(f"   🎬 처리된 프레임: {frame_count}/{total_frames}")
            
            if device == 'cuda':
                theoretical_fps = 1000 / avg_inference
                efficiency = (avg_fps / theoretical_fps) * 100
                print(f"   📊 GPU 효율성: {efficiency:.1f}%")
        
        print(f"\n✅ 동영상 처리 완료!")
        print(f"📁 결과 파일: {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv10을 사용한 동영상 물체인식 (GPU 가속)')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='입력 동영상 파일 경로')
    parser.add_argument('--output', '-o', type=str, 
                       help='출력 동영상 파일 경로 (기본값: input_filename_gpu_detected.mp4)')
    parser.add_argument('--model', '-m', type=str, default='yolov10n.pt',
                       help='YOLO 모델 파일 경로 (기본값: yolov10n.pt)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='신뢰도 임계값 (기본값: 0.5)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='GPU 사용 안함 (CPU만 사용)')
    
    args = parser.parse_args()
    
    # 입력 파일 존재 확인
    if not os.path.exists(args.input):
        print(f"❌ 오류: 입력 파일이 존재하지 않습니다: {args.input}")
        return
    
    # 출력 파일명 설정
    if args.output is None:
        input_path = Path(args.input)
        suffix = "_gpu_detected" if not args.no_gpu and torch.cuda.is_available() else "_cpu_detected"
        output_filename = f"{input_path.stem}{suffix}{input_path.suffix}"
        args.output = str(input_path.parent / output_filename)
    
    use_gpu = not args.no_gpu
    
    print(f"🎯 YOLOv10 동영상 물체인식")
    print(f"📁 입력 파일: {args.input}")
    print(f"📁 출력 파일: {args.output}")
    print(f"🤖 모델: {args.model}")
    print(f"📊 신뢰도 임계값: {args.confidence}")
    print(f"⚡ GPU 사용: {'예' if use_gpu and torch.cuda.is_available() else '아니오'}")
    print("=" * 60)
    
    # 동영상 처리 실행
    process_video_with_yolo(
        input_video_path=args.input,
        output_video_path=args.output,
        model_path=args.model,
        confidence=args.confidence,
        use_gpu=use_gpu
    )

if __name__ == "__main__":
    # 직접 실행 시 예제
    if len(os.sys.argv) == 1:
        print("🚀 YOLOv10 동영상 물체인식 프로그램 (GPU 가속)")
        print("\n📖 사용 예시:")
        print("python yolo_video_detection.py --input video1.mp4")
        print("python yolo_video_detection.py --input video1.mp4 --output result.mp4")
        print("python yolo_video_detection.py --input video1.mp4 --model yolov10s.pt --confidence 0.7")
        print("python yolo_video_detection.py --input video1.mp4 --no-gpu  # CPU만 사용")
        
        # GPU 상태 확인 및 표시
        print("\n" + "=" * 60)
        gpu_available, gpu_info = check_gpu_status()
        
        # 현재 디렉토리의 동영상 파일들 표시
        current_dir = Path(".")
        video_files = list(current_dir.glob("*.mp4")) + list(current_dir.glob("*.avi")) + list(current_dir.glob("*.mov"))
        
        # 원본 동영상 파일만 필터링
        original_videos = []
        for video_file in video_files:
            if not any(keyword in video_file.stem.lower() for keyword in ['detected', 'gpu', 'cpu', 'optimized', 'quick', 'batch']):
                original_videos.append(video_file)
        
        if original_videos:
            print(f"\n📹 현재 디렉토리의 원본 동영상 파일들:")
            for video_file in original_videos:
                print(f"   - {video_file.name}")
            
            # 첫 번째 동영상 파일로 자동 실행
            input_video = str(original_videos[0])
            suffix = "_gpu_demo" if gpu_available else "_cpu_demo"
            output_video = f"{original_videos[0].stem}{suffix}{original_videos[0].suffix}"
            
            print(f"\n🎬 첫 번째 동영상 파일로 GPU 가속 실행: {input_video}")
            process_video_with_yolo(input_video, output_video, use_gpu=True)
        else:
            print("\n❌ 원본 동영상 파일이 발견되지 않았습니다.")
    else:
        main()