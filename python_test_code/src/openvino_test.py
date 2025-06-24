import openvino as ov
import cv2
import numpy as np

# 1. OpenVINO 모델 로딩
core = ov.Core()
compiled_model = core.compile_model(
    "/home/kgh/robot_ws/src/python/openvino_python/model/best.xml", "AUTO"
)

# 입력/출력 정보
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
input_shape = input_layer.shape  # (1, 3, H, W)

# 2. USB 웹캠 열기 (필요시 인덱스 조정)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ USB 웹캠을 열 수 없습니다.")
    exit()

print("✅ 웹캠이 열렸습니다. 실시간 분석 시작...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    orig_h, orig_w = frame.shape[:2]

    # 3. 전처리: resize → RGB → NCHW → float32
    resized = cv2.resize(frame, (input_shape[3], input_shape[2]))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(np.transpose(rgb_image, (2,0,1)), axis=0)
    input_tensor = input_tensor.astype(np.float32) / 255.0

    # 4. 추론
    infer_request = compiled_model.create_infer_request()
    infer_request.set_tensor(input_layer, ov.Tensor(input_tensor))
    infer_request.start_async()
    infer_request.wait()
    output_tensor = infer_request.get_tensor(output_layer)

    # 원본: (1, C, N) → (1, N, C)
    output_data = output_tensor.data.transpose((0, 2, 1))

    # 5. 후처리 및 바운딩 박스 그리기
    # 여기서 C = 5 + 클래스 개수 (YOLOv8n: 클래스 4개 → C = 9)
    for det in output_data[0]:
        # det = [x_center, y_center, w, h, score_cls0, score_cls1, ...]
        bbox = det[0:4]
        class_scores = det[4:]
        class_id = int(np.argmax(class_scores))
        conf = float(class_scores[class_id])

        # 신뢰도 임계치
        if conf < 0.5:
            continue

        x_center, y_center, w, h = bbox
        x_min = int((x_center - w/2) / input_shape[3] * orig_w)
        x_max = int((x_center + w/2) / input_shape[3] * orig_w)
        y_min = int((y_center - h/2) / input_shape[2] * orig_h)
        y_max = int((y_center + h/2) / input_shape[2] * orig_h)

        # 박스와 라벨 그리기
        label = f"{class_id}: {conf:.2f}"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
        cv2.putText(frame, label, (x_min, y_min-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # 6. 결과 영상 표시
    cv2.imshow("OpenVINO YOLOv8n - Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. 종료 처리
cap.release()
cv2.destroyAllWindows()
