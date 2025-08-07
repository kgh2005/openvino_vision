# 🔍 OpenVINO Vision Inference with YOLOv8n and YOLOv11n

ROS2 기반 실시간 객체 탐지 시스템  
YOLO 모델을 OpenVINO IR 형식으로 변환하여, 실시간 카메라 스트림에서 고속 추론을 수행합니다.

---

## 🚀 Overview

- YOLOv8n / YOLOv11n 모델을 OpenVINO로 변환하여 실시간 추론 수행
- ROS2 Humble 기반에서 OpenCV를 활용한 바운딩 박스 시각화
- Python 테스트 코드 포함 (ROS2 외부 환경에서도 실행 가능)

---

## 🖥️ Development Environment

- **OS**: Ubuntu 22.04  
- **Framework**: ROS 2 Humble  
- **OpenVINO Version**: 2025.1  
- **Camera**: `/camera/image_raw`  
- **Model Input Format**: `[1, N, 6]` (NMS=True export)

---

## 📁 Model Preparation

### 1. YOLO → OpenVINO IR 모델 변환

공식 문서 참고: [Ultralytics ↗](https://docs.ultralytics.com/ko/integrations/openvino/)

```bash
# Anaconda 환경에서 실행
yolo export \
  model=/path/to/best.pt \
  format=openvino \
  nms=True
```

> 위 명령어는 `.xml` / `.bin` 파일을 생성합니다.

---

### 2. OpenVINO 설치 (선택)

#### 📦 APT 설치 (권장)

[APT 링크 바로가기 ↗](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?PACKAGE=OPENVINO_BASE&VERSION=v_2025_1_0&OP_SYSTEM=LINUX&DISTRIBUTION=APT)

---

## 📸 실행 방법 (ROS2 기반)

### 1. 카메라 노드 실행

```bash
ros2 launch ocam_ros2 ocam_ros.launch.py 
```

### 2. OpenVINO Vision 노드 실행

```bash
ros2 launch openvino_vision openvino_vision_launch.py
```

---

## 🧪 Python 테스트 코드

- `python_test_code/` 디렉토리 내에 포함
- ROS2 환경 없이 IR 모델 성능을 테스트할 수 있음
- OpenCV 기반으로 추론 결과 시각화 가능

---

## 📌 참고 사항

- 본 프로젝트는 ROS2에서 OpenVINO 추론을 통합한 구조로, 확장성 및 실시간성이 우수합니다.
- `YOLOv8n`이 `YOLOv11n` 대비 인식률 및 속도에서 더 나은 결과를 보였습니다 (주관적 비교).

---

## 🤝 Contributions

Pull requests and issues are welcome.