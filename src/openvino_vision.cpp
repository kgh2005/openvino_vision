// src/openvino_vision.cpp

#include "openvino_vision/openvino_vision.hpp"

OpenVINOVisionNode::OpenVINOVisionNode()
    : Node("openvino_vision_node")
{
  declare_parameter<std::string>("model_xml", "model/best.xml");
  std::string model_path;
  get_parameter("model_xml", model_path);

  try
  {
    // 모델 로드 및 컴파일 (read_model returns shared_ptr<ov::Model>)
    auto model = core_.read_model(model_path);
    compiled_model_ = core_.compile_model(model, "AUTO"); // CPU -> AUTO로 변경
    RCLCPP_INFO(get_logger(), "Loaded model: %s", model_path.c_str());
  }
  catch (const std::exception &e)
  {
    RCLCPP_FATAL(get_logger(), "Model load failed: %s", e.what());
    rclcpp::shutdown();
    return;
  }

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/camera/image_raw", 10,
      std::bind(&OpenVINOVisionNode::imageCallback, this, std::placeholders::_1));
}

void OpenVINOVisionNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  cv::Mat bgr_image = cv_bridge::toCvShare(msg, "bgr8")->image;
  int original_height = bgr_image.rows, original_width = bgr_image.cols;

  auto input_port = compiled_model_.input(0);
  auto output_port = compiled_model_.output(0);
  auto input_shape = input_port.get_shape();
  int model_input_height = static_cast<int>(input_shape[2]);
  int model_input_width = static_cast<int>(input_shape[3]);

  // 전처리: resize → RGB → NCHW → float32 (파이썬 코드와 동일하게)
  cv::Mat resized_image;
  cv::resize(bgr_image, resized_image, cv::Size(model_input_width, model_input_height));
  cv::Mat rgb_image;
  cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

  // float32로 변환하고 정규화
  cv::Mat normalized_image;
  rgb_image.convertTo(normalized_image, CV_32F, 1.0f / 255.0f);

  // NCHW 형태로 변환
  ov::Tensor input_tensor(input_port.get_element_type(), input_shape);
  float *input_data = input_tensor.data<float>();

  std::vector<cv::Mat> color_channels(3);
  cv::split(normalized_image, color_channels);
  int pixels_per_channel = model_input_height * model_input_width;
  for (int channel_idx = 0; channel_idx < 3; ++channel_idx)
  {
    std::memcpy(input_data + channel_idx * pixels_per_channel,
                color_channels[channel_idx].ptr<float>(),
                pixels_per_channel * sizeof(float));
  }

  // 추론 실행 (시간 측정)
  auto start_time = std::chrono::high_resolution_clock::now();

  // 추론 실행
  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_tensor(input_port, input_tensor);
  infer_request.start_async();
  infer_request.wait();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  RCLCPP_INFO(get_logger(), "Inference time: %ld ms", inference_time.count());

  // 출력 처리
  auto output_tensor = infer_request.get_tensor(output_port);
  const float *output_data = output_tensor.data<float>();
  auto output_shape = output_port.get_shape();

  // 파이썬 코드처럼 (1, C, N) → (1, N, C) 변환
  size_t num_classes_plus_coords = output_shape[1]; // 클래스 수 + 4 (bbox)
  size_t num_detections = output_shape[2]; // detection 수

  std::vector<float> transposed_output(num_detections * num_classes_plus_coords);
  for (size_t detection_idx = 0; detection_idx < num_detections; detection_idx++)
  {
    for (size_t class_idx = 0; class_idx < num_classes_plus_coords; class_idx++)
    {
      transposed_output[detection_idx * num_classes_plus_coords + class_idx] =
          output_data[class_idx * num_detections + detection_idx];
    }
  }

  std::vector<DetectionResult> detection_results_;
  float confidence_threshold = 0.5f;

  for (size_t detection_idx = 0; detection_idx < num_detections; detection_idx++)
  {
    const float *detection_data = &transposed_output[detection_idx * num_classes_plus_coords];

    // 바운딩 박스 좌표 (파이썬 코드와 동일)
    float x_center = detection_data[0];
    float y_center = detection_data[1];
    float bbox_width = detection_data[2];
    float bbox_height = detection_data[3];

    // 클래스 점수 (detection_data[4] 이후)
    const float *class_scores = &detection_data[4];
    int best_class_id = 0;
    float max_class_score = class_scores[0];
    size_t num_classes = num_classes_plus_coords - 4;

    for (size_t class_idx = 1; class_idx < num_classes; ++class_idx)
    {
      if (class_scores[class_idx] > max_class_score)
      {
        max_class_score = class_scores[class_idx];
        best_class_id = static_cast<int>(class_idx);
      }
    }

    if (max_class_score < confidence_threshold)
      continue;

    // 좌표 변환 (정규화된 좌표를 원본 이미지 크기로)
    int x_min = static_cast<int>((x_center - bbox_width / 2) / model_input_width * original_width);
    int x_max = static_cast<int>((x_center + bbox_width / 2) / model_input_width * original_width);
    int y_min = static_cast<int>((y_center - bbox_height / 2) / model_input_height * original_height);
    int y_max = static_cast<int>((y_center + bbox_height / 2) / model_input_height * original_height);

    // 경계 체크
    x_min = std::clamp(x_min, 0, original_width - 1);
    x_max = std::clamp(x_max, 0, original_width - 1);
    y_min = std::clamp(y_min, 0, original_height - 1);
    y_max = std::clamp(y_max, 0, original_height - 1);

    detection_results_.push_back({cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min),
                                 max_class_score, best_class_id});
  }

  for (size_t result_idx = 0; result_idx < detection_results_.size(); result_idx++) {
    const auto &detection = detection_results_[result_idx];
    RCLCPP_INFO(get_logger(),
                "Detection %zu: Class=%d, Confidence=%.2f%%, BBox=[%d,%d,%d,%d]",
                result_idx, detection.class_id, detection.confidence * 100,
                detection.bbox.x, detection.bbox.y, detection.bbox.width, detection.bbox.height);
  }

  // 결과 그리기
  for (const auto &detection : detection_results_)
  {
    cv::rectangle(bgr_image, detection.bbox, cv::Scalar(0, 255, 0), 2);
    std::string label = std::to_string(detection.class_id) + ": " +
                        std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
    cv::putText(bgr_image, label, cv::Point(detection.bbox.x, detection.bbox.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
  }

  cv::imshow("OpenVINO YOLOv8n - Webcam", bgr_image);
  cv::waitKey(1);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OpenVINOVisionNode>());
  rclcpp::shutdown();
  return 0;
}
