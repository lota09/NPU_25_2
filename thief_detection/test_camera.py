"""
카메라 테스트 스크립트
사용 가능한 모든 카메라를 찾고 테스트합니다.
"""

import cv2
import sys


def find_cameras(max_test=10):
    """사용 가능한 모든 카메라 찾기"""
    available_cameras = []
    
    print("카메라 검색 중...\n")
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                
                print(f"✅ 카메라 {i} 발견: {width}x{height} @ {fps}fps")
            else:
                print(f"⚠️  카메라 {i}: 열렸지만 프레임을 읽을 수 없음")
            cap.release()
        else:
            # 첫 번째 실패 이후 3개 연속 실패하면 중단
            if i > 0 and len(available_cameras) > 0:
                break
    
    return available_cameras


def test_camera_live(camera_index=0, duration=5):
    """특정 카메라로 실시간 영상 테스트"""
    print(f"\n카메라 {camera_index} 라이브 테스트 시작 ({duration}초)...")
    print("'q'를 눌러 종료")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ 카메라 {camera_index}를 열 수 없습니다.")
        return False
    
    import time
    start_time = time.time()
    frame_count = 0
    
    try:
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            frame_count += 1
            
            # 정보 표시
            cv2.putText(frame, f"Camera {camera_index}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(f"Camera {camera_index} Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n사용자에 의해 중지됨")
                break
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n✅ 테스트 완료:")
        print(f"   - 총 프레임: {frame_count}")
        print(f"   - 실제 FPS: {fps:.2f}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return True


def main():
    print("="*60)
    print("카메라 테스트 도구")
    print("="*60 + "\n")
    
    # 1. 사용 가능한 모든 카메라 찾기
    cameras = find_cameras()
    
    if not cameras:
        print("\n❌ 사용 가능한 카메라를 찾을 수 없습니다.")
        print("\n해결 방법:")
        print("1. USB 카메라가 제대로 연결되었는지 확인")
        print("2. 다른 프로그램에서 카메라를 사용 중인지 확인")
        print("3. Windows 설정 > 개인정보 > 카메라에서 권한 확인")
        return
    
    print(f"\n총 {len(cameras)}개의 카메라를 찾았습니다.\n")
    
    # 2. 사용할 카메라 선택
    if len(cameras) == 1:
        selected = cameras[0]['index']
        print(f"카메라 {selected}을(를) 자동 선택합니다.\n")
    else:
        print("테스트할 카메라를 선택하세요:")
        for cam in cameras:
            print(f"  {cam['index']}. 카메라 {cam['index']} ({cam['width']}x{cam['height']})")
        
        try:
            selected = int(input("\n선택 (번호 입력): "))
            if selected not in [c['index'] for c in cameras]:
                print(f"잘못된 선택입니다. 카메라 0을 사용합니다.")
                selected = 0
        except:
            print("잘못된 입력입니다. 카메라 0을 사용합니다.")
            selected = 0
    
    # 3. 선택한 카메라 테스트
    test_camera_live(selected, duration=10)
    
    print("\n" + "="*60)
    print("config.json 설정 권장값:")
    print(f'  "camera_index": {selected}')
    print("="*60)


if __name__ == "__main__":
    main()
