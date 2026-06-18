import cv2
import os
import glob
import time
from huggingface_hub import hf_hub_download

def get_yunet_model_path():
    print("YuNet ONNX 모델 다운로드/로드 중...")
    model_path = hf_hub_download(
        repo_id="opencv/face_detection_yunet",
        filename="face_detection_yunet_2023mar.onnx"
    )
    return model_path

def evaluate_yunet_detection():
    # 1. 모델 준비
    yunet_model_path = get_yunet_model_path()
    
    # 2. 결과 저장 폴더 생성
    input_dir = 'images'
    output_dir = 'scratch/yunet_eval_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_dir, 'test*.jpg'))
    if not image_paths:
        print(f"'{input_dir}' 폴더에 test*.jpg 이미지가 없습니다.")
        return

    # 설정값 (task.md 요구사항 기준)
    score_threshold = 0.8
    nms_threshold = 0.3
    top_k = 5000
    
    # Detector 초기 인스턴스화 (초기 크기는 임의 지정, 이미지마다 변경)
    detector = cv2.FaceDetectorYN.create(
        model=yunet_model_path,
        config="",
        input_size=(320, 320),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k
    )

    print(f"\n총 {len(image_paths)}개의 테스트 이미지에 대해 평가를 시작합니다...")
    print("-" * 75)
    print(f"{'Filename':<15} | {'Resolution':<15} | {'Time (ms)':<10} | {'Faces (>=0.8)':<15}")
    print("-" * 75)

    total_time = 0.0
    total_faces = 0

    for img_path in sorted(image_paths):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"{filename:<15} | 읽기 실패!")
            continue
            
        h, w, _ = img.shape
        
        # 이미지 크기가 너무 크면 리사이즈 (옵션, 메모리/속도 최적화)
        # max_dim = 1024
        # scale = min(max_dim / w, max_dim / h)
        # if scale < 1.0:
        #     img = cv2.resize(img, (int(w * scale), int(h * scale)))
        #     h, w, _ = img.shape
            
        # 해당 이미지 크기로 input_size 업데이트
        detector.setInputSize((w, h))

        # 추론 시간 측정
        start_time = time.perf_counter()
        ret, faces = detector.detect(img)
        end_time = time.perf_counter()
        
        infer_time_ms = (end_time - start_time) * 1000
        total_time += infer_time_ms
        
        face_count = 0
        if faces is not None:
            face_count = len(faces)
            total_faces += face_count
            
            # 시각화 (Bounding Box 및 Confidence Score 그리기)
            for face in faces:
                box = list(map(int, face[:4]))
                color = (0, 255, 0)
                thickness = 2
                cv2.rectangle(img, box, color, thickness, cv2.LINE_AA)
                
                # Confidence score는 face의 15번째(index 14) 원소입니다.
                confidence = face[14]
                text = f"{confidence:.3f}"
                cv2.putText(img, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # 결과 저장
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, img)

        print(f"{filename:<15} | {str(w)+'x'+str(h):<15} | {infer_time_ms:<10.2f} | {face_count:<15}")

    print("-" * 75)
    avg_time = total_time / len(image_paths) if len(image_paths) > 0 else 0
    print(f"평균 추론 시간: {avg_time:.2f} ms/image")
    print(f"총 검출된 얼굴 수: {total_faces}")
    print(f"시각화 결과는 '{output_dir}'에 저장되었습니다.")

if __name__ == '__main__':
    evaluate_yunet_detection()
