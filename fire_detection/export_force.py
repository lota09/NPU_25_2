"""
YOLOv7 모델의 SiLU를 LeakyReLU로 강제 교체하여 ONNX 변환
DeepX NPU가 지원하는 활성화 함수로만 구성된 모델 생성

NPU 컴파일 요구사항:
1. 활성화 함수: LeakyReLU, ReLU만 지원 (SiLU는 불가)
2. Output: 3개 분리 tensor (병합된 1개는 불가)
3. Batch size: 고정 (동적 batch는 불가)
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy

# YOLOv7 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'yolov7'))

from models.experimental import attempt_load


def replace_activations(model, target_act=nn.SiLU, new_act_cls=nn.LeakyReLU, verbose=True):
    """
    모델 내 모든 target_act(SiLU)를 new_act_cls(LeakyReLU)로 재귀적으로 교체
    
    Args:
        model: 대상 PyTorch 모델
        target_act: 찾을 활성화 함수 클래스 (기본: nn.SiLU)
        new_act_cls: 대체할 활성화 함수 클래스 (기본: nn.LeakyReLU)
        verbose: 로그 출력 여부
    """
    replaced_count = 0
    
    # 재귀적으로 모든 하위 모듈 탐색
    for name, module in model.named_modules():
        if isinstance(module, target_act):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            # 부모 모듈 찾기
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            
            # 활성화 함수 교체
            new_activation = new_act_cls(negative_slope=0.1, inplace=True)
            setattr(parent, child_name, new_activation)
            
            if verbose:
                print(f"✅ 교체됨: {name}")
                print(f"   {target_act.__name__} → {new_act_cls.__name__}")
            
            replaced_count += 1
    
    return replaced_count


def export_force_npu(
    weights_path='models/v7_merged_100epoch_16batch.pt',
    output_name='best_npu.onnx',
    img_size=(640, 640),
    device='cpu',
    opset_version=11
):
    """
    YOLOv7 모델을 NPU 호환 ONNX로 변환
    
    Args:
        weights_path: 학습된 .pt 파일 경로
        output_name: 출력 ONNX 파일명
        img_size: 입력 이미지 크기
        device: 디바이스 (cpu 또는 cuda)
        opset_version: ONNX opset 버전
    """
    
    print("=" * 70)
    print("🚀 YOLOv7 → NPU 호환 ONNX 변환 시작")
    print("=" * 70)
    
    weights_path = Path(weights_path)
    output_path = weights_path.parent / output_name
    
    # 1. 모델 로드
    print(f"\n[1/4] 모델 로드: {weights_path}")
    try:
        model = attempt_load(str(weights_path), map_location=device)
        print(f"✅ 로드 성공 (디바이스: {device})")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return False

    # 2. SiLU → LeakyReLU 강제 교체
    print(f"\n[2/4] SiLU → LeakyReLU 강제 교체")
    replaced = replace_activations(model, target_act=nn.SiLU, new_act_cls=nn.LeakyReLU, verbose=True)
    print(f"✅ SiLU → LeakyReLU 교체 완료 (총 {replaced}개)")
    
    # 3. Export 모드 설정
    print(f"\n[3/4] Export 모드 설정 및 모델 준비")
    
    # Detect 모듈(model.model[-1])의 export 플래그 제거
    # export=False일 때 연결된 단일 출력 (25200,85) 형식 사용
    if hasattr(model.model[-1], 'export'):
        model.model[-1].export = False
        print("✅ Detect.export = False (연결된 단일 출력 형식)")
    else:
        print("⚠️  Detect 모듈에 export 속성이 없습니다")
    
    # 모델을 eval 모드로 변경
    model.eval()
    model.to(device)
    
    # gradient 비활성화
    for p in model.parameters():
        p.requires_grad = False
    
    print("✅ 모델 eval 모드 및 gradient 비활성화")
    
    # 4. ONNX 변환
    print(f"\n[4/4] ONNX 변환 (opset_version={opset_version})")
    
    try:
        # 더미 입력: 원래 형식 NCHW [1, 3, H, W]
        dummy_input = torch.zeros(1, 3, img_size[0], img_size[1], device=device)
        
        # ONNX 변환
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=['images'],
            output_names=['output0'],  # 단일 출력
            dynamic_axes=None,  # NPU 호환성: 배치 크기 고정
            verbose=False,
            opset_version=opset_version,
            do_constant_folding=True,
        )
        
        # 파일 크기 확인
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ ONNX 변환 완료: {output_path}")
        print(f"   파일 크기: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ 모든 단계 완료!")
    print("=" * 70)
    print(f"\n📌 다음 단계:")
    print(f"   1. check_onnx_output.py를 사용하여 출력 구조 확인")
    print(f"      python check_onnx_output.py {output_path}")
    print(f"   2. yolov7_fire.json 설정 파일 준비")
    print(f"   3. dx_com으로 컴파일")
    print(f"      ~/dx_com/dx_com -m {output_name} -c yolov7_fire.json -o ./output_dxnn")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv7 NPU 호환 ONNX 변환')
    parser.add_argument('--weights', type=str, default='models/v7_merged_100epoch_16batch.pt',
                        help='학습된 모델 경로')
    parser.add_argument('--output', type=str, default='best_npu.onnx',
                        help='출력 ONNX 파일명')
    parser.add_argument('--img-size', type=int, nargs=2, default=[640, 640],
                        help='입력 이미지 크기')
    parser.add_argument('--device', type=str, default='cpu',
                        help='디바이스 (cpu 또는 cuda)')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset 버전')
    
    args = parser.parse_args()
    
    success = export_force_npu(
        weights_path=args.weights,
        output_name=args.output,
        img_size=tuple(args.img_size),
        device=args.device,
        opset_version=args.opset
    )
    
    sys.exit(0 if success else 1)
