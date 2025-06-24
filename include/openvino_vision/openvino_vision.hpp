#ifndef OPENVINO_VISION__OPENVINO_VISION_HPP_
#define OPENVINO_VISION__OPENVINO_VISION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <openvino/openvino.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <openvino/openvino.hpp>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>


struct DetectionResult {
    cv::Rect bbox;
    float confidence;
    int class_id;
};

class OpenVINOVisionNode : public rclcpp::Node
{
public:
  OpenVINOVisionNode();

private:
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);


  // OpenVINO 엔진
  ov::Core core_; // OpenVINO 런타임 코어
  std::shared_ptr<ov::Model> model_; // 그래프(모델) 객체
  ov::CompiledModel compiled_model_; // 컴파일된 모델 객체
  ov::InferRequest infer_request_; // 추론 요청 객체

  // ROS 구독자
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
};

#endif  // OPENVINO_VISION__OPENVINO_VISION_HPP_
