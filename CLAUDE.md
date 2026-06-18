# CLAUDE.md

이 파일은 Claude Code(claude.ai/code)가 이 저장소에서 작업할 때 참고하는 안내 문서입니다.

## 환경 설정

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter notebook
```

**평가 스크립트 실행:**
```bash
python yunet_evaluation.py     # YuNet 감지 및 시간 측정 결과 출력
python measure_8vcpu_time.py   # 8 vCPU 환경 벤치마크 시뮬레이션
```

## 아키텍처

이 프로젝트는 여러 모델을 비교 평가하는 얼굴 인식 연구용 파이프라인으로, 세 단계로 구성됩니다.

**1단계 — 얼굴 감지**
- YuNet: `cv2.FaceDetectorYN` 사용 (주요 모델; HuggingFace에서 ONNX 다운로드)
- RetinaFace, Insightface 도 대안으로 사용
- 출력: 얼굴당 바운딩 박스 + 5개 랜드마크

**2단계 — 얼굴 정렬**
- `insightface.utils.face_align.norm_crop()`으로 5개 랜드마크 기준 112×112 정규화
- 대용량 이미지는 감지 전 리사이즈하여 메모리/속도 최적화

**3단계 — 임베딩 및 클러스터링**
- 모든 임베딩 모델은 512차원 L2 정규화 벡터 출력
  - **ArcFace**: `w600k_r50.onnx` (Insightface 제공)
  - **AuraFace**: `notebook/auraface/glintr100.onnx` (`fal/AuraFace-v1`)
  - **AdaFace**: `notebook/adaface/model.safetensors` (`minchul/cvlface_adaface_ir50_webface4m`)
- 클러스터링: 코사인 거리 기반 HDBSCAN
- 시각화: t-SNE/PCA 2D 투영 후 얼굴 썸네일을 데이터 포인트로 오버레이

## 주요 파일

| 파일 | 역할 |
|------|------|
| `notebook/Yunet-detection.ipynb` | 핵심 노트북: 전체 파이프라인 (감지 → 임베딩 → 클러스터링 → 시각화) |
| `notebook/embeddingTest.ipynb` | ArcFace / AdaFace / AuraFace 간 코사인 유사도 비교 |
| `notebook/insightface-detection.ipynb` | Insightface 통합, 68/106점 랜드마크, 성별/나이 모델 |
| `notebook/yolov8n.ipynb` | YOLOv8 nano를 대안 감지 모델로 테스트 |
| `measure_8vcpu_time.py` | ONNX 8스레드 제약 환경에서 전체 파이프라인 성능 측정 |
| `yunet_evaluation.py` | `test*.jpg` 이미지에 YuNet 평가, 결과를 `scratch/yunet_eval_outputs/`에 저장 |

## 구현 시 주의사항

- `requirements.txt`는 **UTF-16LE 인코딩** — 코드에서 파싱 시 인코딩 지정 필요
- ONNX Runtime은 CPU 전용(`onnxruntime`, GPU 버전 아님); `measure_8vcpu_time.py`는 `inter_op_num_threads=8` 명시 설정
- YuNet 신뢰도 임계값: `Yunet-detection.ipynb`는 0.9, `yunet_evaluation.py`는 기본 0.6
- `.gitignore`에 `models/`, `weights/`, `*.onnx`, `*.pt`, `*.safetensors`, `images/` 제외 — 모델 가중치와 데이터셋은 커밋하지 않음
- 모델은 런타임에 `huggingface_hub.hf_hub_download()`로 자동 다운로드

## 데이터셋 구조

```
images/
  child/        # 어린이 얼굴 이미지 55장 (주요 벤치마크 데이터셋)
  similar*/     # 임베딩 유사도 테스트용 소규모 세트
  different*/   # 서로 다른 인물 세트
  test*.jpg     # yunet_evaluation.py용 번호 붙은 이미지 16장
train/          # 학습 참조용 신원 폴더 300개 이상 (n000002–n000502)
```
