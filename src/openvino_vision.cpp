// src/openvino_vision.cpp

#include "openvino_vision/openvino_vision.hpp"

OpenVINOVisionNode::OpenVINOVisionNode()
    : Node("openvino_vision_node")
{
  // declare_parameter<std::string>("model_xml", "model/best.xml");
  declare_parameter<std::string>("model_xml");
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

void OpenVINOVisionNode::imageProcessing()
{
  if (bgr_image.empty())
    return;

  int original_height = bgr_image.rows;
  int original_width = bgr_image.cols;

  // ---- 1. 입력 이미지 전처리 ----
  auto input_port = compiled_model_.input(0);
  auto input_shape = input_port.get_shape();
  int model_input_height = static_cast<int>(input_shape[2]);
  int model_input_width = static_cast<int>(input_shape[3]);

  cv::Mat resized_image;
  cv::resize(bgr_image, resized_image, cv::Size(model_input_width, model_input_height));
  cv::Mat rgb_image;
  cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

  cv::Mat normalized_image;
  rgb_image.convertTo(normalized_image, CV_32F, 1.0f / 255.0f);

  ov::Tensor input_tensor(input_port.get_element_type(), input_shape);
  float *input_data = input_tensor.data<float>();
  std::vector<cv::Mat> color_channels(3);
  cv::split(normalized_image, color_channels);
  int pixels_per_channel = model_input_height * model_input_width;
  for (int c = 0; c < 3; ++c)
  {
    std::memcpy(input_data + c * pixels_per_channel,
                color_channels[c].ptr<float>(),
                pixels_per_channel * sizeof(float));
  }

  // ---- 2. 모델 추론 ----
  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_tensor(input_port, input_tensor);
  infer_request.start_async();
  infer_request.wait();

  auto output_tensor = infer_request.get_tensor(compiled_model_.output(0));
  const float *output_data = output_tensor.data<float>();
  auto shape = output_tensor.get_shape(); // [1, N, 6] - NMS=True 출력
  size_t num_detections = shape[1];

  // ---- 3. detection 후처리 ----
  for (size_t detection_idx = 0; detection_idx < num_detections; detection_idx++)
  {
    // NMS=True 출력: [x1, y1, x2, y2, confidence, class_id]
    const float *det = &output_data[detection_idx * 6];

    float x1 = det[0];
    float y1 = det[1];
    float x2 = det[2];
    float y2 = det[3];
    float confidence = det[4];
    int class_id = static_cast<int>(det[5]);

    int bx1, by1, bx2, by2;

    // 픽셀 좌표를 입력 해상도 기준으로 스케일링
    bx1 = static_cast<int>(x1 / model_input_width * original_width);
    by1 = static_cast<int>(y1 / model_input_height * original_height);
    bx2 = static_cast<int>(x2 / model_input_width * original_width);
    by2 = static_cast<int>(y2 / model_input_height * original_height);

    // 좌표 클리핑
    bx1 = std::clamp(bx1, 0, original_width - 1);
    by1 = std::clamp(by1, 0, original_height - 1);
    bx2 = std::clamp(bx2, 0, original_width - 1);
    by2 = std::clamp(by2, 0, original_height - 1);

    // 유효한 바운딩 박스인지 확인
    if (bx2 <= bx1 || by2 <= by1)
    {
      continue;
    }

    float threshold = 0.5f;
    if (confidence < threshold)
    {
      continue;
    }

    // 시각화
    cv::Point pt1(bx1, by1);
    cv::Point pt2(bx2, by2);

    // 바운딩 박스 그리기
    cv::rectangle(bgr_image, cv::Rect(pt1, pt2), cv::Scalar(0, 255, 0), 2);

    // 텍스트 정보 만들기
    std::string label = "class_" + std::to_string(class_id) + " (" + std::to_string(static_cast<int>(confidence * 100)) + "%)";

    // 텍스트 위치: 바운딩 박스 왼쪽 위
    int baseline = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::Point label_origin(pt1.x, std::max(pt1.y - 5, label_size.height));

    // 배경 박스 (선택)
    cv::rectangle(bgr_image,
                  cv::Rect(label_origin.x, label_origin.y - label_size.height, label_size.width, label_size.height + baseline),
                  cv::Scalar(0, 255, 0), cv::FILLED);

    // 텍스트 그리기
    cv::putText(bgr_image, label, label_origin,
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  }

  cv::imshow("OpenVINO", bgr_image);
  cv::waitKey(1);
}

void OpenVINOVisionNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  try
  {
    bgr_image = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    imageProcessing();
  }
  catch (const cv_bridge::Exception &e)
  {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
  }
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OpenVINOVisionNode>());
  rclcpp::shutdown();
  return 0;
}
