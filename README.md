# openvino

### 결과

영상 ⇒ result_video 폴더

주관적 생각 ⇒ yolov8n이 object detection 성능이 더 좋았음.

- yolov8n

⇒ 평균 속도 11ms

- yolov11n

⇒ 평균 속도 13ms

### 환경

- ubuntu 22.04
- ros2 humble

### test 모델

- yolov8n
- yolov11n

### pytorch ⇒ IR 파일로 변경

https://docs.ultralytics.com/ko/integrations/openvino/

⇒ 참고

anaconda 환경에서 진행함.

```cpp
# IR파일로 변환
yolo export model=yolo11n.pt format=openvino # creates 'yolo11n_openvino_model/'

yolo export model=C:\Users\msi\ultralytics\runs\detect\train4yolov8n\weights\best.pt format=openvino
```

https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?PACKAGE=OPENVINO_BASE&VERSION=v_2025_1_0&OP_SYSTEM=LINUX&DISTRIBUTION=APT

pip와 apt 다운로드 시 참고

### 기타 참고사항

- python_test_code 파일에 있는 코드는 ros2  환경이 아닌 환경에서 테스트하기 위한 코드
- 실행 명령어 (카메라 관련 노드는 따로 켜 야 함.)

```c
ros2 launch openvino_vision openvino_vision_launch.py
```
