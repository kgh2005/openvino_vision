import openvino as ov
import cv2
import numpy as np

# 1. OpenVINO 모델 로딩
core = ov.Core()
compiled_model = core.compile_model(
    "/home/robit/test/model/yolo11n.xml", "AUTO"
)

# 입력/출력 정보
input_layer = compiled_model.input(0)
input_shape = input_layer.shape  # (1, 3, H, W)
input_h, input_w = input_shape[2], input_shape[3]

print(f"Input shape: {input_shape}")
print(f"Number of outputs: {len(compiled_model.outputs)}")

# 출력 레이어들 확인
outputs = {}
for i, output in enumerate(compiled_model.outputs):
    print(f"Output {i}: {output.any_name} - {output.shape}")
    outputs[output.any_name] = output

# COCO 클래스 이름들
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# 2. USB 웹캠 열기
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
    resized = cv2.resize(frame, (input_w, input_h))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(np.transpose(rgb_image, (2, 0, 1)), axis=0)
    input_tensor = input_tensor.astype(np.float32) / 255.0

    # 4. 추론
    infer_request = compiled_model.create_infer_request()
    infer_request.set_tensor(input_layer, ov.Tensor(input_tensor))
    infer_request.start_async()
    infer_request.wait()

    # 5. NMS=True 모델 출력 처리 ([1,300,6] 형태)
    if len(compiled_model.outputs) == 1:
        output_tensor = infer_request.get_tensor(compiled_model.output(0))
        detections = output_tensor.data
        
        # 배치 차원 제거: [1,300,6] -> [300,6]
        if len(detections.shape) == 3:
            detections = detections[0]
        
        detection_count = 0
        for detection in detections:
            if len(detection) >= 6:
                x1, y1, x2, y2, confidence, class_id = detection[:6]
                
                # 신뢰도 확인
                if confidence < 0.3:  # 임계값을 낮춰서 테스트
                    continue
                
                detection_count += 1
                
                # 좌표가 정규화되어 있는지 확인
                if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                    # 정규화된 좌표 -> 원본 이미지 좌표
                    x1 = int(x1 * orig_w)
                    y1 = int(y1 * orig_h)
                    x2 = int(x2 * orig_w)
                    y2 = int(y2 * orig_h)
                else:
                    # 이미 픽셀 좌표인 경우, 입력 해상도 기준으로 스케일링
                    x1 = int(x1 / input_w * orig_w)
                    y1 = int(y1 / input_h * orig_h)
                    x2 = int(x2 / input_w * orig_w)
                    y2 = int(y2 / input_h * orig_h)
                
                # 좌표 클리핑
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))
                
                # 유효한 바운딩 박스인지 확인
                if x2 <= x1 or y2 <= y1:
                    continue
                
                class_id = int(class_id)
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                label = f"{class_name}: {confidence:.2f}"
                
                # 디버그 정보 (첫 5개만)
                if detection_count <= 5:
                    print(f"Detection {detection_count}: {class_name}, conf={confidence:.3f}, box=({x1},{y1},{x2},{y2})")
                
                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 검출 개수 표시
        if detection_count > 0:
            info_text = f"Detections: {detection_count}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    elif len(compiled_model.outputs) >= 3:
        # 다중 출력: boxes, scores, classes 분리
        try:
            # 일반적인 출력 이름들
            possible_box_names = ['boxes', 'bboxes', 'detection_boxes', 'output0']
            possible_score_names = ['scores', 'confidences', 'detection_scores', 'output1']
            possible_class_names = ['classes', 'labels', 'detection_classes', 'output2']
            
            boxes_tensor = None
            scores_tensor = None
            classes_tensor = None
            
            # 출력 텐서 찾기
            for name in possible_box_names:
                if name in outputs:
                    boxes_tensor = infer_request.get_tensor(outputs[name])
                    break
            
            for name in possible_score_names:
                if name in outputs:
                    scores_tensor = infer_request.get_tensor(outputs[name])
                    break
            
            for name in possible_class_names:
                if name in outputs:
                    classes_tensor = infer_request.get_tensor(outputs[name])
                    break
            
            # 인덱스로 접근 (이름으로 찾지 못한 경우)
            if boxes_tensor is None:
                boxes_tensor = infer_request.get_tensor(compiled_model.output(0))
            if scores_tensor is None and len(compiled_model.outputs) > 1:
                scores_tensor = infer_request.get_tensor(compiled_model.output(1))
            if classes_tensor is None and len(compiled_model.outputs) > 2:
                classes_tensor = infer_request.get_tensor(compiled_model.output(2))
            
            boxes_data = boxes_tensor.data
            scores_data = scores_tensor.data if scores_tensor is not None else None
            classes_data = classes_tensor.data if classes_tensor is not None else None
            
            # 배치 차원 제거
            if len(boxes_data.shape) > 2:
                boxes_data = boxes_data[0]
            if scores_data is not None and len(scores_data.shape) > 1:
                scores_data = scores_data[0]
            if classes_data is not None and len(classes_data.shape) > 1:
                classes_data = classes_data[0]
            
            # 검출 결과 처리
            num_detections = len(boxes_data)
            if scores_data is not None:
                num_detections = min(num_detections, len(scores_data))
            if classes_data is not None:
                num_detections = min(num_detections, len(classes_data))
            
            for i in range(num_detections):
                confidence = scores_data[i] if scores_data is not None else 1.0
                
                if confidence < 0.5:
                    continue
                
                # 바운딩 박스 좌표
                if len(boxes_data[i]) >= 4:
                    x1, y1, x2, y2 = boxes_data[i][:4]
                    
                    # 좌표 변환
                    x1 = int(x1 * orig_w)
                    y1 = int(y1 * orig_h)
                    x2 = int(x2 * orig_w)
                    y2 = int(y2 * orig_h)
                    
                    # 좌표 클리핑
                    x1 = max(0, min(x1, orig_w))
                    y1 = max(0, min(y1, orig_h))
                    x2 = max(0, min(x2, orig_w))
                    y2 = max(0, min(y2, orig_h))
                    
                    # 클래스 정보
                    class_id = int(classes_data[i]) if classes_data is not None else 0
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        except Exception as e:
            print(f"다중 출력 처리 중 오류: {e}")
            # 첫 번째 출력만 사용하는 fallback
            output_tensor = infer_request.get_tensor(compiled_model.output(0))
            detections = output_tensor.data
            print(f"Fallback output shape: {detections.shape}")

    # 6. 결과 영상 표시
    cv2.imshow("YOLOv11n + OpenVINO (NMS=True)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. 종료 처리
cap.release()
cv2.destroyAllWindows()
