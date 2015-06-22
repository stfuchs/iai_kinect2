/**
 * Copyright 2014 University of Bremen, Institute for Artificial Intelligence
 * Author: Thiemo Wiedemeyer <wiedemeyer@cs.uni-bremen.de>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/SetCameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/image_encodings.h>

#include <tf/transform_broadcaster.h>

#include <compressed_depth_image_transport/compression_common.h>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/config.h>

#include <kinect2_bridge/kinect2_definitions.h>
#include <kinect2_bridge/kinect2_conversions.h>

class Kinect2BridgeRaw
{
private:
  std::vector<int> compression_params_;
  std::string compression_16_bit_ext_, compression_16_bit_string_, base_name_tf_;

  cv::Size size_rgb_, size_ir_;
  cv::Mat rgb_, ir_, depth_;

  std::vector<std::thread> threads_;
  std::mutex lock_ir_depth_, lock_rgb_;
  std::mutex lock_sync_, lock_pub_, lock_time_, lock_status_;

  bool publish_tf_;
  std::thread tf_thread_, main_thread_;

  libfreenect2::Freenect2 freenect2_;
  libfreenect2::Freenect2Device *device_;
  libfreenect2::SyncMultiFrameListener *listener_rgb_, *listener_ir_depth_;
  libfreenect2::PacketPipeline *packet_pipeline_;
  std::shared_ptr<libfreenect2::Frame> ir_frame_, depth_frame_, rgb_frame_;

  ros::NodeHandle nh_, priv_nh_;

  size_t frame_rgb_, frame_ir_depth_, pub_frame_rgb_, pub_frame_ir_depth_;
  ros::Time last_rgb_, last_depth_;

  bool next_rgb_, next_ir_depth_;
  double delta_t_, elapsed_time_rgb_, elapsed_time_ir_depth_;
  bool running_, device_active_, client_connected_;

  enum Image
  {
    IR_SD = 0,
    DEPTH_SD,
    COLOR_HD,
    COUNT
  };

  enum Status
  {
    UNSUBCRIBED = 0,
    RAW,
    COMPRESSED,
    BOTH
  };

  std::vector<ros::Publisher> image_pubs_, compressed_pubs_;
  ros::Publisher info_rgb_pub_, info_ir_pub_;
  sensor_msgs::CameraInfo info_rgb_, info_ir_;
  std::vector<Status> status_;

public:
  Kinect2Bridge(const ros::NodeHandle &nh = ros::NodeHandle(),
                const ros::NodeHandle &priv_nh = ros::NodeHandle("~"))
    : size_rgb_(1920, 1080)
    , size_ir_(512, 424)
    , nh_(nh)
    , priv_nh_(priv_nh)
    , frame_rgb_(0)
    , frame_ir_depth_(0)
    , pub_frame_rgb_(0)
    , pub_frame_ir_depth_(0)
    , last_rgb_(0, 0)
    , last_depth_(0, 0)
    , next_rgb_(false)
    , next_ir_depth_(false)
    , running(false)
    , device_active_(false)
    , client_connected_(false)
  {
    rgb_ = cv::Mat::zeros(size_rgb_, CV_8UC3);
    ir_ = cv::Mat::zeros(size_ir_, CV_32F);
    depth_ = cv::Mat::zeros(size_ir_, CV_32F);
    status_.resize(COUNT, UNSUBCRIBED);
  }

  void start()
  {
    if(!initialize()) { return; }
    running = true;

    if(publish_tf_)
    {
      tf_thread_ = std::thread(&Kinect2Bridge::publishStaticTF, this);
    }

    for(size_t i = 0; i < threads.size(); ++i)
    {
      threads[i] = std::thread(&Kinect2Bridge::threadDispatcher, this, i);
    }

    main_thread_ = std::thread(&Kinect2Bridge::main, this);
  }

  void stop()
  {
    running = false;
    main_thread_.join();

    for(size_t i = 0; i < threads.size(); ++i) { threads[i].join(); }

    if(publish_tf_) { tf_thread_.join(); }

    device->stop();
    device->close();
    delete listener_ir_depth_;
    delete listener_rgb_;

    for(size_t i = 0; i < COUNT; ++i)
    {
      image_pubs_[i].shutdown();
      compressed_pubs_[i].shutdown();
      info_rgb_pub_.shutdown();
      info_ir_pub_.shutdown();
    }

    nh.shutdown();
  }

private:
  bool initialize()
  {
    double fps_limit, max_depth, min_depth;
    bool use_png, bilateral_filter, edge_aware_filter;
    int32_t jpeg_quality, png_level, queue_size, depth_dev, worker_threads;
    std::string depth_method, sensor, base_name;

    std::string dept_default = "cpu";

#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
    depthDefault = "opengl";
#endif
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
    depthDefault = "opencl";
#endif

    priv_nh.param("base_name", base_name, std::string(K2_DEFAULT_NS));
    priv_nh.param("sensor", sensor, std::string(""));
    priv_nh.param("fps_limit", fps_limit, -1.0);
    priv_nh.param("use_png", use_png, false);
    priv_nh.param("jpeg_quality", jpeg_quality, 90);
    priv_nh.param("png_level", png_level, 1);
    priv_nh.param("depth_method", depth_method, depth_default);
    priv_nh.param("depth_device", depth_dev, -1);
    priv_nh.param("max_depth", max_depth, 12.0);
    priv_nh.param("min_depth", min_depth, 0.1);
    priv_nh.param("queue_size", queue_size, 2);
    priv_nh.param("bilateral_filter", bilateral_filter, true);
    priv_nh.param("edge_aware_filter", edge_aware_filter, true);
    priv_nh.param("publish_tf", publish_tf, false);
    priv_nh.param("base_name_tf", base_name_tf, base_name);
    priv_nh.param("worker_threads", worker_threads, 2);

    worker_threads = std::max(1, worker_threads);
    threads.resize(worker_threads);

    std::cout << "parameter:" << std::endl <<
      "        base_name: " << base_name << std::endl <<
      "           sensor: " << sensor << std::endl <<
      "        fps_limit: " << fps_limit << std::endl <<
      "          use_png: " << (use_png ? "true" : "false") << std::endl <<
      "     jpeg_quality: " << jpeg_quality << std::endl <<
      "        png_level: " << png_level << std::endl <<
      "     depth_method: " << depth_method << std::endl <<
      "     depth_device: " << depth_dev << std::endl <<
      "        max_depth: " << max_depth << std::endl <<
      "        min_depth: " << min_depth << std::endl <<
      "       queue_size: " << queueSize << std::endl <<
      " bilateral_filter: " << (bilateral_filter ? "true" : "false") << std::endl <<
      "edge_aware_filter: " << (edge_aware_filter ? "true" : "false") << std::endl <<
      "       publish_tf: " << (publish_tf ? "true" : "false") << std::endl <<
      "     base_name_tf: " << base_name_tf << std::endl <<
      "   worker_threads: " << worker_threads << std::endl << std::endl;

    delta_t_ = fps_limit > 0 ? 1.0 / fps_limit : 0.0;

    initCompression(jpeg_quality, png_level, use_png);

    bool ret = true;
    ret = ret && initPipeline(depth_method, depth_dev,
                              bilateral_filter, edge_aware_filter,
                              min_depth, max_depth);
    ret = ret && initDevice(sensor);
    initTopics(queue_size, base_name);

    return ret;
  }

  void initCompression(const int32_t jpeg_quality, const int32_t png_level,
                       const bool use_png)
  {
    compression_params_ = { CV_IMWRITE_JPEG_QUALITY, jpeg_quality, CV_IMWRITE_PNG_COMPRESSION,
                            png_level, CV_IMWRITE_PNG_STRATEGY, CV_IMWRITE_PNG_STRATEGY_RLE, 0 };

    if(use_png)
    {
      compression_16_bit_ext_ = ".png";
      compression_16_bit_string_ = sensor_msgs::image_encodings::TYPE_16UC1 + "; png compressed";
    }
    else
    {
      compression_16_bit_ext_ = ".tif";
      compression_16_bit_string_ = sensor_msgs::image_encodings::TYPE_16UC1 + "; tiff compressed";
    }
  }

  bool initPipeline(const std::string &method, const int32_t device,
                    const bool bilateral_filter, const bool edge_aware_filter,
                    const double minDepth, const double maxDepth)
  {
    if(method == "default")
    {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      packet_pipeline_ = new libfreenect2::OpenCLPacketPipeline(device);
#elif defined(LIBFREENECT2_WITH_OPENGL_SUPPORT)
      packet_pipeline_ = new libfreenect2::OpenGLPacketPipeline();
#else
      packet_pipeline_ = new libfreenect2::CpuPacketPipeline();
#endif
    }
    else if(method == "cpu")
    {
      packet_pipeline_ = new libfreenect2::CpuPacketPipeline();
    }
    else if(method == "opencl")
    {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      packet_pipeline_ = new libfreenect2::OpenCLPacketPipeline(device);
#else
      std::cerr << "OpenCL depth processing is not available!" << std::endl;
      return false;
#endif
    }
    else if(method == "opengl")
    {
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
      packet_pipeline_ = new libfreenect2::OpenGLPacketPipeline();
#else
      std::cerr << "OpenGL depth processing is not available!" << std::endl;
      return false;
#endif
    }
    else
    {
      std::cerr << "Unknown depth processing method: " << method << std::endl;
      return false;
    }

    libfreenect2::DepthPacketProcessor::Config config;
    config.EnableBilateralFilter = bilateral_filter;
    config.EnableEdgeAwareFilter = edge_aware_filter;
    config.MinDepth = minDepth;
    config.MaxDepth = maxDepth;
    packet_pipeline_->getDepthPacketProcessor()->setConfiguration(config);
    return true;
  }
  
  bool initDevice(std::string &sensor)
  {
    bool deviceFound = false;
    const int numOfDevs = freenect2.enumerateDevices();

    if(numOfDevs <= 0)
    {
      std::cerr << "Error: no Kinect2 devices found!" << std::endl;
      return false;
    }

    if(sensor.empty())
    {
      sensor = freenect2.getDefaultDeviceSerialNumber();
    }

    std::cout << "Kinect2 devices found: " << std::endl;
    for(int i = 0; i < numOfDevs; ++i)
    {
      const std::string &s = freenect2.getDeviceSerialNumber(i);
      deviceFound = deviceFound || s == sensor;
      std::cout << "  " << i << ": " << s << (s == sensor ? " (selected)" : "") << std::endl;
    }

    if(!deviceFound)
    {
      std::cerr << "Error: Device with serial '" << sensor << "' not found!" << std::endl;
      return false;
    }

    device = freenect2.openDevice(sensor, packet_pipeline_);

    if(device == 0)
    {
      std::cout << "no device connected or failure opening the default one!" << std::endl;
      return -1;
    }

    listener_rgb_ = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color);
    listener_ir_depth_ = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);

    device->setColorFrameListener(listener_rgb_);
    device->setIrAndDepthFrameListener(listener_ir_depth_);

    std::cout << std::endl << "starting kinect2" << std::endl << std::endl;
    device->start();

    std::cout << std::endl << "device serial: " << sensor << std::endl;
    std::cout << "device firmware: " << device->getFirmwareVersion() << std::endl;

    libfreenect2::Freenect2Device::ColorCameraParams rgb_params
      = device->getColorCameraParams();
    libfreenect2::Freenect2Device::IrCameraParams ir_params
      = device->getIrCameraParams();

    device->stop();

    std::cout << std::endl << "default ir camera parameters: " << std::endl;
    std::cout <<   "fx " << ir_params.fx
              << ", fy " << ir_params.fy
              << ", cx " << ir_params.cx
              << ", cy " << ir_params.cy << std::endl;
    std::cout <<   "k1 " << ir_params.k1
              << ", k2 " << ir_params.k2
              << ", p1 " << ir_params.p1
              << ", p2 " << ir_params.p2
              << ", k3 " << ir_params.k3 << std::endl;

    std::cout << std::endl << "default rgb camera parameters: " << std::endl;
    std::cout <<   "fx " << rgb_params.fx
              << ", fy " << rgb_params.fy
              << ", cx " << rgb_params.cx
              << ", cy " << rgb_params.cy << std::endl;
    toCameraInfo(rgb_params, info_rgb_);
    toCameraInfo(ir_params, info_ir_);
    return true;
  }

  
  void initTopics(const int32_t queue_size, const std::string &base_name)
  {
    std::vector<std::string> topics(COUNT);
    topics[IR_SD]    = K2_TOPIC_SD K2_TOPIC_IMAGE_IR;
    topics[DEPTH_SD] = K2_TOPIC_SD K2_TOPIC_IMAGE_DEPTH;
    topics[COLOR_HD] = K2_TOPIC_HD K2_TOPIC_IMAGE_COLOR;

    image_pubs_.resize(COUNT);
    compressed_pubs_.resize(COUNT);
    ros::SubscriberStatusCallback cb = boost::bind(&Kinect2Bridge::callbackStatus, this);

    for(size_t i = 0; i < COUNT; ++i)
    {
      image_pubs_[i] = nh.advertise<sensor_msgs::Image>(
        base_name + topics[i], queue_size, cb, cb);
      compressed_pubs_[i] = nh.advertise<sensor_msgs::CompressedImage>(
        base_name + topics[i] + K2_TOPIC_COMPRESSED, queue_size, cb, cb);
    }
    info_rgb_pub_ = nh.advertise<sensor_msgs::CameraInfo>(
      base_name + K2_TOPIC_HD + K2_TOPIC_INFO, queue_size, cb, cb);
    info_ir_pub_ = nh.advertise<sensor_msgs::CameraInfo>(
      base_name + K2_TOPIC_SD + K2_TOPIC_INFO, queue_size, cb, cb);
  }

  void callbackStatus()
  {
    lock_status_.lock();
    client_connected_ = updateStatus();

    if(client_connected_ && !device_active_)
    {
      std::cout << "[kinect2_bridge] client connected. starting device..."
                << std::endl << std::flush;
      device_active_ = true;
      device->start();
    }
    else if(!client_connected_ && device_active_)
    {
      std::cout << "[kinect2_bridge] no clients connected. stopping device..."
                << std::endl << std::flush;
      device_active_ = false;
      device->stop();
    }
    lock_status_.unlock();
  }

  bool updateStatus()
  {
    bool any = false;
    for(size_t i = 0; i < COUNT; ++i)
    {
      Status s = UNSUBCRIBED;
      if(image_pubs_[i].getNumSubscribers() > 0)
        s = RAW;
      if(compressed_pubs_[i].getNumSubscribers() > 0)
        s = (s == RAW ? BOTH : COMPRESSED);

      status[i] = s;
      any = (any || s != UNSUBCRIBED);
    }
    return any || info_rgb_pub_.getNumSubscribers()>0 || info_ir_pub_.getNumSubscribers()>0;
  }

  void main()
  {
    std::cout << "[kinect2_bridge] waiting for clients to connect"
              << std::endl << std::endl;
    double next_frame = ros::Time::now().toSec() + delta_t_;
    double fps_time = ros::Time::now().toSec();
    size_t old_frame_ir_depth = 0, old_frame_rgb = 0;
    next_rgb_ = true;
    next_ir_depth_ = true;

    for(; running_ && ros::ok();)
    {
      if(!device_active_)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        fps_time =  ros::Time::now().toSec();
        next_frame = fps_time + delta_t_;
        continue;
      }

      double now = ros::Time::now().toSec();

      if(now - fps_time >= 3.0)
      {
        fps_time = now - fps_time;
        size_t frames_ir_depth = frame_ir_depth_ - old_frame_ir_depth;
        size_t frames_rgb = frame_rgb_ - old_frame_rgb;
        old_frame_ir_depth = frame_ir_depth_;
        old_frame_rgb = frame_rgb_;

        lock_time_.lock();
        double t_color = elapsed_time_rgb_;
        double t_depth = elapsed_time_ir_depth_;
        elapsed_time_rgb_ = 0;
        elapsed_time_ir_depth_ = 0;
        lock_time_.unlock();

        std::cout << "[kinect2_bridge] depth processing: ~"
                  << frames_ir_depth / t_depth << "Hz ("
                  << (t_depth / frames_ir_depth) * 1000 << "ms) publishing rate: ~"
                  << frames_ir_depth / fps_time << "Hz" << std::endl
                  << "[kinect2_bridge] color processing: ~"
                  << frames_rgb / t_color << "Hz ("
                  << (t_color / frames_rgb) * 1000 << "ms) publishing rate: ~"
                  << frames_rgb / fps_time << "Hz" << std::endl << std::flush;
        fps_time = now;
      }

      if(now >= next_frame)
      {
        next_rgb_ = true;
        next_ir_depth_ = true;
        next_frame += delta_t_;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(10));

      if(!device_active_)
      {
        lock_time_.lock();
        elapsed_time_rgb_ = 0;
        elapsed_time_ir_depth_ = 0;
        lock_time_.unlock();
        continue;
      }
    }
  }

  void threadDispatcher(const size_t id)
  {
    const size_t checkFirst = id % 2;
    bool processedFrame = false;
    int oldNice = nice(0);
    oldNice = nice(19 - oldNice);

    for(; running_ && ros::ok();)
    {
      processedFrame = false;

      for(size_t i = 0; i < 2; ++i)
      {
        if(i == checkFirst)
        {
          if(next_ir_depth_ && lock_ir_depth_.try_lock())
          {
            next_ir_depth_ = false;
            receiveIrDepth();
            processedFrame = true;
          }
        }
        else
        {
          if(next_rgb_ && lock_rgb_.try_lock())
          {
            next_rgb_ = false;
            receiveColor();
            processedFrame = true;
          }
        }
      }

      if(!processedFrame)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

  void receiveIrDepth()
  {
    libfreenect2::FrameMap frames;
    if(!receiveFrames(listener_ir_depth_, frames))
    {
      lock_ir_depth_.unlock();
      return;
    }
    double now = ros::Time::now().toSec();

    ir_frame_ = std::shared_ptr<libfreenect2::Frame>(frames[libfreenect2::Frame::Ir]);
    depth_frame_ = std::shared_ptr<libfreenect2::Frame>(frames[libfreenect2::Frame::Depth]);
    cv::Mat ir(ir_frame_->height, ir_frame_->width, CV_32FC1, ir_frame_->data);
    cv::Mat depth(depth_frame_->height, depth_frame_->width, CV_32FC1, depth_frame_->data);

    size_t frame = frame_ir_depth_++;
    lock_ir_depth_.unlock();

    std::vector<Status> status = this->status_; // thread safety?
    std::vector<cv::Mat> images(COUNT);
    if(status[IR_SD])
    {
      ir.convertTo(images[IR_SD], CV_16U);
      //cv::flip(images[IR_SD], images[IR_SD], 1);
    }

    if(status[DEPTH_SD])
    {
      depth.convertTo(images[DEPTH_SD], CV_16U);
      //cv::flip(images[DEPTH_SD], images[DEPTH_SD], 1);
    }

    std_msgs::Header header = createHeader(last_depth_, last_rgb_);
    publishImages(images, header, status, frame, pub_frame_ir_depth_, IR_SD, COLOR_HD);

    double elapsed = ros::Time::now().toSec() - now;
    lock_time_.lock();
    elapsed_time_ir_depth_ += elapsed;
    lock_time_.unlock();
  }

  void receiveColor()
  {
    libfreenect2::FrameMap frames;
    if(!receiveFrames(listener_rgb_, frames))
    {
      lock_rgb_.unlock();
      return;
    }
    double now = ros::Time::now().toSec();

    rgb_frame_ = std::shared_ptr<libfreenect2::Frame>(frames[libfreenect2::Frame::Color]);
    cv::Mat rgb(rgb_frame_->height, rgb_frame_->width, CV_8UC4, rgb_frame_->data);

    size_t frame = frame_rgb_++;
    lock_rgb_.unlock();

    std::vector<Status> status = this->status_;
    std::vector<cv::Mat> images(COUNT);
    if(status[COLOR_HD])
    {
      //cv::Mat tmp;
      //cv::flip(rgb, tmp, 1);
      cv::cvtColor(rgb, images[COLOR_HD], CV_BGRA2BGR);
    }

    std_msgs::Header header = createHeader(last_rgb_, last_depth_);
    publishImages(images, header, status, frame, pub_frame_rgb_, COLOR_HD, COUNT);

    double elapsed = ros::Time::now().toSec() - now;
    lock_time_.lock();
    elapsed_time_rgb_ += elapsed;
    lock_time_.unlock();
  }

  bool receiveFrames(libfreenect2::SyncMultiFrameListener *listener, libfreenect2::FrameMap &frames)
  {
    bool newFrames = false;
    for(; !newFrames;)
    {
#ifdef LIBFREENECT2_THREADING_STDLIB
      newFrames = listener->waitForNewFrame(frames, 1000);
#else
      newFrames = true;
      listener->waitForNewFrame(frames);
#endif
      if(!device_active_ || !running || !ros::ok())
      {
        if(newFrames)
        {
          listener->release(frames);
        }
        return false;
      }
    }
    return true;
  }

  std_msgs::Header createHeader(ros::Time &last, ros::Time &other)
  {
    ros::Time timestamp = ros::Time::now();
    lock_sync_.lock();
    if(other.isZero())
    {
      last = timestamp;
    }
    else
    {
      timestamp = other;
      other = ros::Time(0, 0);
    }
    lock_sync_.unlock();

    std_msgs::Header header;
    header.seq = 0;
    header.stamp = timestamp;
    header.frame_id = K2_TF_RGB_OPT_FRAME;
    return header;
  }

  void publishImages(const std::vector<cv::Mat> &images,
                     const std_msgs::Header &header,
                     const std::vector<Status> &status,
                     const size_t frame,
                     size_t &pubFrame,
                     const size_t begin, const size_t end)
  {
    std::vector<sensor_msgs::ImagePtr> imageMsgs(COUNT);
    std::vector<sensor_msgs::CompressedImagePtr> compressedMsgs(COUNT);
    sensor_msgs::CameraInfoPtr infoHDMsg, infoIRMsg;
    std_msgs::Header _header = header;

    if(begin < COLOR_HD)
    {
      _header.frame_id = base_name_tf_ + K2_TF_IR_OPT_FRAME;

      infoIRMsg = sensor_msgs::CameraInfoPtr(new sensor_msgs::CameraInfo);
      *infoIRMsg = infoIR;
      infoIRMsg->header = _header;
    }
    else
    {
      _header.frame_id = base_name_tf_ + K2_TF_RGB_OPT_FRAME;

      infoHDMsg = sensor_msgs::CameraInfoPtr(new sensor_msgs::CameraInfo);
      *infoHDMsg = infoHD;
      infoHDMsg->header = _header;
    }

    for(size_t i = begin; i < end; ++i)
    {
      if(i < DEPTH_HD || i == COLOR_SD_RECT)
      {
        _header.frame_id = base_name_tf_ + K2_TF_IR_OPT_FRAME;
      }
      else
      {
        _header.frame_id = base_name_tf_ + K2_TF_RGB_OPT_FRAME;
      }

      switch(status[i])
      {
      case UNSUBCRIBED:
        break;
      case RAW:
        imageMsgs[i] = sensor_msgs::ImagePtr(new sensor_msgs::Image);
        createImage(images[i], _header, Image(i), *imageMsgs[i]);
        break;
      case COMPRESSED:
        compressedMsgs[i] = sensor_msgs::CompressedImagePtr(new sensor_msgs::CompressedImage);
        createCompressed(images[i], _header, Image(i), *compressedMsgs[i]);
        break;
      case BOTH:
        imageMsgs[i] = sensor_msgs::ImagePtr(new sensor_msgs::Image);
        compressedMsgs[i] = sensor_msgs::CompressedImagePtr(new sensor_msgs::CompressedImage);
        createImage(images[i], _header, Image(i), *imageMsgs[i]);
        createCompressed(images[i], _header, Image(i), *compressedMsgs[i]);
        break;
      }
    }

    while(frame != pubFrame)
    {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    lock_pub_.lock();
    for(size_t i = begin; i < end; ++i)
    {
      switch(status[i])
      {
      case UNSUBCRIBED:
        break;
      case RAW:
        image_pubs_[i].publish(imageMsgs[i]);
        break;
      case COMPRESSED:
        compressed_pubs_[i].publish(compressedMsgs[i]);
        break;
      case BOTH:
        image_pubs_[i].publish(imageMsgs[i]);
        compressed_pubs_[i].publish(compressedMsgs[i]);
        break;
      }
    }

    if(begin < COLOR_HD)
    {
      if(infoIRPub.getNumSubscribers() > 0)
      {
        infoIRPub.publish(infoIRMsg);
      }
    }
    else
    {
      if(infoHDPub.getNumSubscribers() > 0)
      {
        infoHDPub.publish(infoHDMsg);
      }
      if(infoQHDPub.getNumSubscribers() > 0)
      {
        infoQHDPub.publish(infoQHDMsg);
      }
    }

    ++pubFrame;
    lock_pub_.unlock();
  }

  void createImage(const cv::Mat &image, const std_msgs::Header &header, const Image type, sensor_msgs::Image &msgImage) const
  {
    size_t step, size;
    step = image.cols * image.elemSize();
    size = image.rows * step;

    switch(type)
    {
    case IR_SD:
    case IR_SD_RECT:
    case DEPTH_SD:
    case DEPTH_SD_RECT:
    case DEPTH_HD:
    case DEPTH_QHD:
      msgImage.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
      break;
    case COLOR_SD_RECT:
    case COLOR_HD:
    case COLOR_HD_RECT:
    case COLOR_QHD:
    case COLOR_QHD_RECT:
      msgImage.encoding = sensor_msgs::image_encodings::BGR8;
      break;
    case MONO_HD:
    case MONO_HD_RECT:
    case MONO_QHD:
    case MONO_QHD_RECT:
      msgImage.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
      break;
    case COUNT:
      return;
    }

    msgImage.header = header;
    msgImage.height = image.rows;
    msgImage.width = image.cols;
    msgImage.is_bigendian = false;
    msgImage.step = step;
    msgImage.data.resize(size);
    memcpy(msgImage.data.data(), image.data, size);
  }

  void createCompressed(const cv::Mat &image, const std_msgs::Header &header, const Image type, sensor_msgs::CompressedImage &msgImage) const
  {
    msgImage.header = header;

    switch(type)
    {
    case IR_SD:
    case IR_SD_RECT:
    case DEPTH_SD:
    case DEPTH_SD_RECT:
    case DEPTH_HD:
    case DEPTH_QHD:
      msgImage.format = compression_16_bit_string_;
      cv::imencode(compression_16_bit_ext_, image, msgImage.data, compression_params_);
      break;
    case COLOR_SD_RECT:
    case COLOR_HD:
    case COLOR_HD_RECT:
    case COLOR_QHD:
    case COLOR_QHD_RECT:
      msgImage.format = sensor_msgs::image_encodings::BGR8 + "; jpeg compressed bgr8";
      cv::imencode(".jpg", image, msgImage.data, compression_params_);
      break;
    case MONO_HD:
    case MONO_HD_RECT:
    case MONO_QHD:
    case MONO_QHD_RECT:
      msgImage.format = sensor_msgs::image_encodings::TYPE_8UC1 + "; jpeg compressed ";
      cv::imencode(".jpg", image, msgImage.data, compression_params_);
      break;
    case COUNT:
      return;
    }
  }

  void publishStaticTF()
  {
    tf::TransformBroadcaster broadcaster;
    tf::StampedTransform stColorOpt, stIrOpt;
    ros::Time now = ros::Time::now();

    tf::Matrix3x3 rot(rotation.at<double>(0, 0), rotation.at<double>(0, 1), rotation.at<double>(0, 2),
                      rotation.at<double>(1, 0), rotation.at<double>(1, 1), rotation.at<double>(1, 2),
                      rotation.at<double>(2, 0), rotation.at<double>(2, 1), rotation.at<double>(2, 2));

    tf::Quaternion qZero;
    qZero.setRPY(0, 0, 0);
    tf::Vector3 trans(translation.at<double>(0), translation.at<double>(1), translation.at<double>(2));
    tf::Vector3 vZero(0, 0, 0);
    tf::Transform tIr(rot, trans), tZero(qZero, vZero);

    stColorOpt = tf::StampedTransform(tZero, now, base_name_tf_ + K2_TF_LINK, base_name_tf_ + K2_TF_RGB_OPT_FRAME);
    stIrOpt = tf::StampedTransform(tIr, now, base_name_tf_ + K2_TF_RGB_OPT_FRAME, base_name_tf_ + K2_TF_IR_OPT_FRAME);

    for(; running && ros::ok();)
    {
      now = ros::Time::now();
      stColorOpt.stamp_ = now;
      stIrOpt.stamp_ = now;

      broadcaster.sendTransform(stColorOpt);
      broadcaster.sendTransform(stIrOpt);

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
};

class Kinect2BridgeNodelet : public nodelet::Nodelet
{
private:
  std::thread kinect2BridgeThread;
  Kinect2Bridge *pKinect2Bridge;

public:
  Kinect2BridgeNodelet() : Nodelet(), pKinect2Bridge(NULL)
  {
  }

  ~Kinect2BridgeNodelet()
  {
    if(pKinect2Bridge)
    {
      pKinect2Bridge->stop();
      delete pKinect2Bridge;
    }
  }

  virtual void onInit()
  {
    pKinect2Bridge = new Kinect2Bridge(getNodeHandle(), getPrivateNodeHandle());
    pKinect2Bridge->start();
  }
};

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(Kinect2BridgeNodelet, nodelet::Nodelet)

void helpOption(const std::string &name, const std::string &stype, const std::string &value, const std::string &desc)
{
  std::cout  << '_' << name << ":=<" << stype << '>' << std::endl
             << "    default: " << value << std::endl
             << "    info:    " << desc << std::endl;
}

void help(const std::string &path)
{
  std::string depthMethods = "cpu";
  std::string depthDefault = "cpu";

#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
  depthMethods += ", opengl";
  depthDefault = "opengl";
#endif
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
  depthMethods += ", opencl";
  depthDefault = "opencl";
#endif

  std::cout << path << " [_options:=value]" << std::endl;
  helpOption("base_name",         "string", K2_DEFAULT_NS,  "set base name for all topics");
  helpOption("sensor",            "double", "-1.0",         "serial of the sensor to use");
  helpOption("fps_limit",         "double", "-1.0",         "limit the frames per second");
  helpOption("calib_path",        "string", K2_CALIB_PATH,  "path to the calibration files");
  helpOption("use_png",           "bool",   "false",        "Use PNG compression instead of TIFF");
  helpOption("jpeg_quality",      "int",    "90",           "JPEG quality level from 0 to 100");
  helpOption("png_level",         "int",    "1",            "PNG compression level from 0 to 9");
  helpOption("depth_method",      "string", depthDefault,   "Use specific depth processing: " + depthMethods);
  helpOption("depth_device",      "int",    "-1",           "openCL device to use for depth processing");
  helpOption("max_depth",         "double", "12.0",         "max depth value");
  helpOption("min_depth",         "double", "0.1",          "min depth value");
  helpOption("queue_size",        "int",    "2",            "queue size of publisher");
  helpOption("bilateral_filter",  "bool",   "true",         "enable bilateral filtering of depth images");
  helpOption("edge_aware_filter", "bool",   "true",         "enable edge aware filtering of depth images");
  helpOption("publish_tf",        "bool",   "false",        "publish static tf transforms for camera");
  helpOption("base_name_tf",      "string", "as base_name", "base name for the tf frames");
  helpOption("worker_threads",    "int",    "4",            "number of threads used for processing the images");
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "kinect2_bridge");


  for(int argI = 1; argI < argc; ++argI)
  {
    std::string arg(argv[argI]);

    if(arg == "--help" || arg == "--h" || arg == "-h" || arg == "-?" || arg == "--?")
    {
      help(argv[0]);
      ros::shutdown();
      return 0;
    }
    else
    {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return -1;
    }
  }

  if(!ros::ok())
  {
    std::cerr << "ros::ok failed!" << std::endl;
    return -1;
  }

  Kinect2Bridge kinect2;
  kinect2.start();

  ros::spin();

  kinect2.stop();

  ros::shutdown();
  return 0;
}
