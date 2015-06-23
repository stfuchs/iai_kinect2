#include <algorithm>
#include <sensor_msgs/CameraInfo.h>
#include <libfreenect2/libfreenect2.hpp>

void toFreenect2(const sensor_msgs::CameraInfo& msg,
                  libfreenect2::Freenect2Device::ColorCameraParams& params)
{
}

void toFreenect2(const sensor_msgs::CameraInfo& msg,
                  libfreenect2::Freenect2Device::IrCameraParams& params)
{
}

void toCameraInfo(const libfreenect2::Freenect2Device::ColorCameraParams& params,
                    sensor_msgs::CameraInfo& msg)
{
  msg.height = 1080;
  msg.width = 1920;
  msg.distortion_model = "kinect2_special";

  msg.D = { 
    params.shift_d,
    params.shift_m,

    params.mx_x3y0, // xxx
    params.mx_x0y3, // yyy
    params.mx_x2y1, // xxy
    params.mx_x1y2, // yyx
    params.mx_x2y0, // xx
    params.mx_x0y2, // yy
    params.mx_x1y1, // xy
    params.mx_x1y0, // x
    params.mx_x0y1, // y
    params.mx_x0y0, // 1

    params.my_x3y0, // xxx
    params.my_x0y3, // yyy
    params.my_x2y1, // xxy
    params.my_x1y2, // yyx
    params.my_x2y0, // xx
    params.my_x0y2, // yy
    params.my_x1y1, // xy
    params.my_x1y0, // x
    params.my_x0y1, // y
    params.my_x0y0 // 1
  };

  msg.K = {{ params.fx, 0, params.cx, 0, params.fy, 0, 0, 0, 1. }};
  msg.R = {{ 1., 0, 0, 0, 1., 0, 0, 0, 1. }};
  msg.P = {{ params.fx, 0, params.cx, -0.0520, 0, params.fy, 0, 0, 0, 0, 1., 0 }};
}

void toCameraInfo(const libfreenect2::Freenect2Device::IrCameraParams& params,
                  sensor_msgs::CameraInfo& msg)
{
  msg.height = 424;
  msg.width = 512;
  msg.distortion_model = "plumb_bob";

  msg.D = { params.k1, params.k2, params.p1, params.p2, params.k3 };
  msg.K = {{ params.fx, 0, params.cx, 0, params.fy, 0, 0, 0, 1. }};
  msg.R = {{ 1., 0, 0, 0, 1., 0, 0, 0, 1. }};
  msg.P = {{ params.fx, 0, params.cx, -0.0520, 0, params.fy, 0, 0, 0, 0, 1., 0 }};
}

void toCameraInfo(const cv::Size& size,
                  const cv::Mat& intrinsic,
                  const cv::Mat& distortion,
                  const cv::Mat& rotation,
                  const cv::Mat& projection,
                  sensor_msgs::CameraInfo& msg)
{
  msg.height = size.height;
  msg.width = size.width;
  msg.distortion_model = "plumb_bob";
  std::copy(distortion.begin<double>(), distortion.end<double>(), msg.D.begin());
  std::copy(intrinsic.begin<double>(), intrinsic.end<double>(), msg.K.begin());
  std::copy(rotation.begin<double>(), rotation.end<double>(), msg.R.begin());
  std::copy(projection.begin<double>(), projection.end<double>(), msg.P.begin());
}
