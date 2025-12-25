"""
YOLOv7 화재 감지 모델 동영상 비교 평가 스크립트
두 개의 모델을 동일한 테스트 동영상으로 평가하여 공정하게 비교
결과 동영상을 생성하여 육안으로 확인 가능
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import time
import sys

# YOLOv7 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'yolov7'))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from utils.datasets import letterbox

def check_gpu_status():
    """GPU 상태 확인"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"✅ CUDA: {torch.version.cuda}")
        return True
    else:
        print("⚠️  GPU 사용 불가 - CPU 모드")
        return False

def load_yolov7_model(weights_path, device):
    """YOLOv7 모델 로드"""
    print(f"🤖 모델 로딩: {weights_path}")
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    print(f"   ✅ 모델 로드 완료")
    
    # GPU 최적화
    if device.type != 'cpu':
        model.half()  # FP16
        print(f"   ✅ FP16 최적화 적용")
        
    return model

def process_video_yolov7(video_path, model, device, output_path, model_name, conf_thres=0.5, iou_thres=0.45, img_size=640, class_conf_thres=None):
    """YOLOv7로 동영상 처리 및 결과 저장
    
    Args:
        class_conf_thres: 클래스별 신뢰도 임계값 dict (예: {'flame': 0.7, 'smoke': 0.3})
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ 동영상 열기 실패: {video_path}")
        return None
    
    # 동영상 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   📺 {width}x{height} | {fps} FPS | {total_frames} frames")
    
    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # 결과 저장용
    detections = []
    inference_times = []
    frame_count = 0
    
    half = device.type != 'cpu'
    
    # 클래스 이름 (데이터셋 설정과 일치: 0=flame, 1=smoke)
    names = ['flame', 'smoke']
    
    # 색상 (flame: 빨강, smoke: 회색)
    colors = [(0, 0, 255), (128, 128, 128)]
    
    # 클래스별 신뢰도 임계값 설정
    if class_conf_thres is None:
        class_conf_thres = {'flame': 0.5, 'smoke': 0.5}  # 기본값: 불꽃 엄격, 연기 관대
    print(f"   🎯 클래스별 임계값: flame={class_conf_thres.get('flame', conf_thres)}, smoke={class_conf_thres.get('smoke', conf_thres)}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 전처리 (종횡비 유지하며 리사이즈 - letterbox)
            img = letterbox(frame, img_size, stride=32, auto=True)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # 추론
            t0 = time.time()
            with torch.no_grad():
                pred = model(img)[0]
            inference_time = (time.time() - t0) * 1000  # ms로 변환
            inference_times.append(inference_time)
            
            # NMS 적용
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            
            # 감지 결과 저장 및 시각화
            det = pred[0]
            if len(det):
                # 좌표 스케일 조정
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                
                # 클래스별 신뢰도 필터링 적용
                filtered_det = []
                for *xyxy, conf, cls in det:
                    cls_idx = int(cls)
                    cls_name = names[cls_idx]
                    cls_threshold = class_conf_thres.get(cls_name, conf_thres)
                    
                    # 클래스별 임계값 통과한 것만 유지
                    if conf >= cls_threshold:
                        filtered_det.append([*xyxy, conf, cls])
                
                # 필터링된 결과로 대체
                if filtered_det:
                    det = torch.tensor(filtered_det).to(det.device)
                    
                    detections.append({
                        'frame': frame_count,
                        'count': len(det),
                        'confidences': det[:, 4].cpu().numpy()
                    })
                    
                    # 바운딩 박스 그리기
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=2)
            
            # 모델 이름과 프레임 정보 추가
            cv2.putText(frame, f'Model: {model_name}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if len(det) if isinstance(det, torch.Tensor) else False:
                cv2.putText(frame, f'Detections: {len(det)}', (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 프레임 저장
            out.write(frame)
            
            # 진행률 표시
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_inference = np.mean(inference_times[-30:])
                print(f"   📊 {progress:.1f}% | ⚡ {avg_inference:.1f}ms", end='\r')
        
        print()  # 줄바꿈
        
    finally:
        cap.release()
        out.release()
    
    # 통계 계산
    stats = {
        'total_frames': total_frames,
        'processed_frames': frame_count,
        'avg_inference_ms': np.mean(inference_times),
        'total_detections': len(detections),
        'avg_confidence': np.mean([np.mean(d['confidences']) for d in detections]) if detections else 0,
        'detection_rate': len(detections) / frame_count * 100 if frame_count > 0 else 0
    }
    
    return stats

def compare_models_on_videos(model1_path, model2_path, video_paths, output_dir='results/model_comparison'):
    """두 모델을 여러 동영상으로 비교 및 결과 동영상 생성"""
    
    print("🔥 YOLOv7 화재 감지 모델 비교 평가")
    print("=" * 70)
    
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU 확인
    gpu_available = check_gpu_status()
    device = select_device('0' if gpu_available else 'cpu')
    
    # 모델 로드
    print(f"\n📦 모델 1 로딩...")
    model1 = load_yolov7_model(model1_path, device)
    model1_name = Path(model1_path).parent.parent.name
    
    print(f"\n📦 모델 2 로딩...")
    model2 = load_yolov7_model(model2_path, device)
    model2_name = Path(model2_path).parent.parent.name
    
    print(f"\n" + "=" * 70)
    print(f"🆚 모델 비교: {model1_name} vs {model2_name}")
    print("=" * 70)
    
    # 각 동영상으로 평가
    results = {
        'model1': {'name': model1_name, 'videos': {}},
        'model2': {'name': model2_name, 'videos': {}}
    }
    
    for video_path in video_paths:
        video_name = Path(video_path).name
        video_stem = Path(video_path).stem
        print(f"\n🎬 테스트 동영상: {video_name}")
        print("-" * 70)
        
        # 출력 경로 설정
        output1 = output_dir / f"{video_stem}_{model1_name}.mp4"
        output2 = output_dir / f"{video_stem}_{model2_name}.mp4"
        
        # 모델 1 평가 및 동영상 생성
        if output1.exists():
            print(f"   ⏭️  {model1_name} 결과 동영상이 이미 존재합니다. 건너뜁니다: {output1.name}")
            stats1 = None
        else:
            print(f"   ⚡ {model1_name} 평가 및 동영상 생성 중...")
            stats1 = process_video_yolov7(video_path, model1, device, output1, model1_name)
            print(f"   💾 저장: {output1}")
        results['model1']['videos'][video_name] = stats1
        
        # 모델 2 평가 및 동영상 생성
        if output2.exists():
            print(f"   ⏭️  {model2_name} 결과 동영상이 이미 존재합니다. 건너뜁니다: {output2.name}")
            stats2 = None
        else:
            print(f"   ⚡ {model2_name} 평가 및 동영상 생성 중...")
            stats2 = process_video_yolov7(video_path, model2, device, output2, model2_name)
            print(f"   💾 저장: {output2}")
        results['model2']['videos'][video_name] = stats2
        
        # 동영상별 비교 출력 (둘 다 처리된 경우만)
        if stats1 and stats2:
            print(f"\n   📊 {video_name} 결과:")
            print(f"   {'항목':<20} {model1_name:<25} {model2_name:<25}")
            print(f"   {'-'*20} {'-'*25} {'-'*25}")
            print(f"   {'추론 시간 (ms)':<20} {stats1['avg_inference_ms']:<25.1f} {stats2['avg_inference_ms']:<25.1f}")
            print(f"   {'총 감지 횟수':<20} {stats1['total_detections']:<25} {stats2['total_detections']:<25}")
            print(f"   {'감지율 (%)':<20} {stats1['detection_rate']:<25.1f} {stats2['detection_rate']:<25.1f}")
            print(f"   {'평균 신뢰도':<20} {stats1['avg_confidence']:<25.3f} {stats2['avg_confidence']:<25.3f}")
        elif stats1 or stats2:
            print(f"\n   ℹ️  {video_name}: 일부 결과가 건너뛰어졌습니다.")
    
    # 전체 요약
    print("\n" + "=" * 70)
    # 평균 계산 (None이 아닌 값만 사용)
    model1_stats = [s for s in results['model1']['videos'].values() if s is not None]
    model2_stats = [s for s in results['model2']['videos'].values() if s is not None]
    
    if not model1_stats and not model2_stats:
        print("\n⚠️  모든 결과가 이미 존재하여 새로 처리된 동영상이 없습니다.")
        print(f"📁 기존 결과 확인: {output_dir.absolute()}")
        return results
    
    model1_avg_inference = np.mean([s['avg_inference_ms'] for s in model1_stats]) if model1_stats else 0
    model2_avg_inference = np.mean([s['avg_inference_ms'] for s in model2_stats]) if model2_stats else 0
    
    model1_total_det = sum([s['total_detections'] for s in model1_stats]) if model1_stats else 0
    model2_total_det = sum([s['total_detections'] for s in model2_stats]) if model2_stats else 0
    
    model1_avg_conf = np.mean([s['avg_confidence'] for s in model1_stats if s['avg_confidence'] > 0]) if model1_stats else 0
    model2_avg_conf = np.mean([s['avg_confidence'] for s in model2_stats if s['avg_confidence'] > 0]) if model2_stats else 0
    
    model1_avg_conf = np.mean([s['avg_confidence'] for s in results['model1']['videos'].values() if s['avg_confidence'] > 0])
    model2_avg_conf = np.mean([s['avg_confidence'] for s in results['model2']['videos'].values() if s['avg_confidence'] > 0])
    
    print(f"\n{'지표':<25} {model1_name:<25} {model2_name:<25}")
    print(f"{'-'*25} {'-'*25} {'-'*25}")
    print(f"{'평균 추론 시간 (ms)':<25} {model1_avg_inference:<25.1f} {model2_avg_inference:<25.1f}")
    print(f"{'총 감지 횟수':<25} {model1_total_det:<25} {model2_total_det:<25}")
    print(f"{'평균 신뢰도':<25} {model1_avg_conf:<25.3f} {model2_avg_conf:<25.3f}")
    
    # 승자 판정
    print(f"\n{'='*70}")
    print("🏆 종합 평가")
    print(f"{'='*70}")
    
    winner_speed = model1_name if model1_avg_inference < model2_avg_inference else model2_name
    winner_detection = model1_name if model1_total_det > model2_total_det else model2_name
    winner_confidence = model1_name if model1_avg_conf > model2_avg_conf else model2_name
    
    print(f"⚡ 속도 우위: {winner_speed}")
    print(f"🎯 감지 성능 우위: {winner_detection} (단, 정답 라벨 없어 참고만)")
    print(f"✅ 신뢰도 우위: {winner_confidence}")
    
    print(f"\n📁 결과 동영상 저장 위치: {output_dir.absolute()}")
    print(f"   - 각 테스트 동영상마다 2개의 결과 파일 생성됨")
    print(f"   - 총 {len(video_paths) * 2}개의 결과 동영상")
    print(f"\n💡 정량적 평가는 runs/*/results.txt의 mAP 값을 참고하세요")
    
    return results

def main():
    """메인 함수"""
    
    # 모델 경로
    model1_path = "runs/v7_merged_100epoch_16batch/weights/best.pt"
    model2_path = "runs/v7_merged_200epoch_16batch/weights/best.pt"
    
    # 테스트 동영상 경로
    video_paths = [
        "assets/bucket11.mp4",
        "assets/printer31.mp4",
        "assets/roomfire41.mp4"
    ]
    
    # 존재 확인
    if not Path(model1_path).exists():
        print(f"❌ 모델 1을 찾을 수 없습니다: {model1_path}")
        return
    
    if not Path(model2_path).exists():
        print(f"❌ 모델 2를 찾을 수 없습니다: {model2_path}")
        return
    
    missing_videos = [v for v in video_paths if not Path(v).exists()]
    if missing_videos:
        print(f"❌ 동영상 파일을 찾을 수 없습니다:")
        for v in missing_videos:
            print(f"   - {v}")
        return
    
    # 비교 실행
    results = compare_models_on_videos(model1_path, model2_path, video_paths)
    
    print(f"\n✅ 비교 평가 완료!")
    print(f"\n📺 결과 동영상을 열어서 육안으로 성능을 비교하세요:")
    print(f"   results/model_comparison/ 폴더에 6개의 동영상 파일")

if __name__ == "__main__":
    main()
