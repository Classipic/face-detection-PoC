import cv2
import time
import os
import glob
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from sklearn.cluster import HDBSCAN ## 어떻게 동작하는지 , numpy로 동작가능한지, 다른 방식은 뭐가 있는지


# ArcFace 표준 정렬 기준점 (112x112)
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def _umeyama(src, dst):
    """Umeyama 유사변환(scale 포함) 추정 — skimage SimilarityTransform.estimate 와 동일."""
    num, dim = src.shape  # (5, 2)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.full((dim + 1, dim + 1), np.nan)
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T


def norm_crop(img, landmark, image_size=112):
    """5개 랜드마크를 ArcFace 기준점에 맞춰 image_size x image_size 로 정렬."""
    M = _umeyama(landmark.astype(np.float64), _ARCFACE_DST)[0:2, :]
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)


def run_benchmark_8vcpu():
    print("Loading models... (Simulating 8 vCPU)")
    
    # Yunet 모델
    model_path = hf_hub_download(
        repo_id="opencv/face_detection_yunet",
        filename="face_detection_yunet_2023mar.onnx"
    )
    
    aura_model_path = "notebook/auraface/glintr100.onnx"
    
    # 핵심: ONNX Runtime의 스레드 수를 8개(8 vCPU 환경 시뮬레이션)로 제한
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 8
    sess_opts.inter_op_num_threads = 8
    
    aura_session = ort.InferenceSession(
        aura_model_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"]
    )
    input_name = aura_session.get_inputs()[0].name
    output_name = aura_session.get_outputs()[0].name

    image_paths = glob.glob('images/child/*.png')
    print(f"Found {len(image_paths)} images.")

    total_detect_time = 0.0
    total_embed_time = 0.0
    total_faces = 0
    total_images_processed = 0
    aura_embeddings = []

    print("-" * 65)
    print(f"{'Image':<15} | {'Faces':<5} | {'Detect (ms)':<15} | {'Embed Total (ms)':<15}")
    print("-" * 65)

    pipeline_start_time = time.perf_counter()

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None: continue
        
        h, w = img.shape[:2]
        
        # Detector 초기화 (cv2.FaceDetectorYN 내부적으로도 OpenCV 스레드풀 사용)
        cv2.setNumThreads(8) # OpenCV 스레드 수 8개로 제한
        detector = cv2.FaceDetectorYN.create(
            model=model_path, config="", input_size=(w, h),
            score_threshold=0.6, nms_threshold=0.3, top_k=5000
        )
        
        # Detect time
        t0 = time.perf_counter()
        _, faces = detector.detect(img)
        t1 = time.perf_counter()
        
        detect_ms = (t1 - t0) * 1000
        total_detect_time += detect_ms
        total_images_processed += 1
        
        embed_ms_total = 0.0
        face_count = 0
        if faces is not None:
            face_count = len(faces)
            total_faces += face_count
            
            # Embed time
            t2 = time.perf_counter()
            for face in faces:
                landmarks = face[4:14].reshape(5, 2)
                aligned_img = norm_crop(img, landmarks)
                aligned_rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
                
                input_blob = np.transpose(aligned_rgb, (2, 0, 1))
                input_blob = np.expand_dims(input_blob, axis=0).astype(np.float32)
                input_blob = (input_blob - 127.5) / 127.5
                
                outputs = aura_session.run([output_name], {input_name: input_blob})
                raw_embedding = outputs[0][0]
                
                norm = np.linalg.norm(raw_embedding)
                normed_embedding = raw_embedding / norm if norm > 0 else raw_embedding
                aura_embeddings.append(normed_embedding)
                
            t3 = time.perf_counter()
            embed_ms_total = (t3 - t2) * 1000
            total_embed_time += embed_ms_total
            
        print(f"{os.path.basename(image_path):<15} | {face_count:<5} | {detect_ms:<15.2f} | {embed_ms_total:<15.2f}")

    print("-" * 65)

    # Clustering Time
    cluster_start_time = time.perf_counter()
    if len(aura_embeddings) > 0:
        embeddings_array = np.array(aura_embeddings)
        hdbscan = HDBSCAN(
            min_cluster_size=2, min_samples=2, metric='cosine', cluster_selection_epsilon=0.15
        )
        # HDBSCAN 내부 스레드 사용(가능한 경우) 제한
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["OPENBLAS_NUM_THREADS"] = "8"
        os.environ["MKL_NUM_THREADS"] = "8"
        cluster_labels = hdbscan.fit_predict(embeddings_array)
        
    cluster_end_time = time.perf_counter()
    cluster_time_ms = (cluster_end_time - cluster_start_time) * 1000
    
    pipeline_end_time = time.perf_counter()
    total_pipeline_time_ms = (pipeline_end_time - pipeline_start_time) * 1000

    print("\n--- 8 vCPU 시뮬레이션 파이프라인 요약 ---")
    print(f"Total Images: {total_images_processed}")
    print(f"Total Faces: {total_faces}")
    print(f"Total Detection Time: {total_detect_time:.2f} ms")
    print(f"Total Embedding Time: {total_embed_time:.2f} ms")
    print(f"Total Clustering Time: {cluster_time_ms:.2f} ms")
    print(f"Total Pipeline Time: {total_pipeline_time_ms:.2f} ms ({total_pipeline_time_ms/1000:.2f} s)")

if __name__ == '__main__':
    run_benchmark_8vcpu()
