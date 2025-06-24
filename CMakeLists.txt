# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(openvino_vision)

# 컴파일러 경고 설정
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS 2 및 의존성
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

# 실행 파일
add_executable(openvino_vision_node src/openvino_vision.cpp)

# 인클루드 디렉토리 (현대적 방식)
target_include_directories(openvino_vision_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# C++17 표준 설정
target_compile_features(openvino_vision_node PUBLIC cxx_std_17)

# ROS 2 의존성
ament_target_dependencies(openvino_vision_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
)

# 라이브러리 링크
target_link_libraries(openvino_vision_node
  openvino::runtime
  ${OpenCV_LIBS}
)

# 설치
install(TARGETS openvino_vision_node
  DESTINATION lib/${PROJECT_NAME})

# 헤더 파일 설치 (include 폴더가 있는 경우)
install(DIRECTORY include/
  DESTINATION include/
  FILES_MATCHING PATTERN "*.hpp")

# 모델 및 런치 파일 설치
install(DIRECTORY model
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
