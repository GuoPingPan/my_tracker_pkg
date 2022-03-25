#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dirent.h>
#include <string>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include "geometry_msgs/Twist.h"
#include "track_pkg/Target.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

static const std::string RGB_WINDOW = "RGB Image window";
//static const std::string DEPTH_WINDOW = "DEPTH Image window";



cv::Mat rgbimage;
cv::Mat depthimage;
cv::Rect selectRect;
cv::Point origin;
cv::Rect result;

bool select_flag = false;
bool bRenewROI = false;  // the flag to enable the implementation of KCF algorithm for the new chosen ROI
bool bBeginKCF = false;
bool enable_get_depth = false;
bool HOG = true;

bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool SILENT = true;
bool LAB = false;

// Create KCFTracker object
KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

float dist_val[5] ;
// float dist_x[3];
float dist_y;

float dist_x;

void onMouse(int event, int x, int y, int, void*)
{
    if (select_flag)
    {
        selectRect.x = MIN(origin.x, x);        
        selectRect.y = MIN(origin.y, y);
        selectRect.width = abs(x - origin.x);   
        selectRect.height = abs(y - origin.y);
        selectRect &= cv::Rect(0, 0, rgbimage.cols, rgbimage.rows);
    }
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        bBeginKCF = false;  
        select_flag = true; 
        origin = cv::Point(x, y);       
        selectRect = cv::Rect(x, y, 0, 0);  
    }
    else if (event == CV_EVENT_LBUTTONUP)
    {
        select_flag = false;
        bRenewROI = true;
    }
}

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Subscriber depth_sub_;

public:
  ros::Publisher pub;
  std::string color_topic;
  std::string depth_topic;
  float fx,fy,cx,cy;
  float scale;
  ImageConverter(std::string& color_image,std::string& depth_image)
    : it_(nh_),color_topic(color_image),depth_topic(depth_image)
  {
    fx = 609.2713012695312;
    fy = 608.010498046875;
    cx = 316.67022705078125;
    cy = 244.8178253173828; 
    scale = 1000;
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe(color_topic, 1, 
      &ImageConverter::imageCb, this);
    depth_sub_ = it_.subscribe(depth_topic, 1, 
      &ImageConverter::depthCb, this);
    pub = nh_.advertise<track_pkg::Target>("tracker/target", 1000);

    cv::namedWindow(RGB_WINDOW);
    //cv::namedWindow(DEPTH_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(RGB_WINDOW);
    //cv::destroyWindow(DEPTH_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv_ptr->image.copyTo(rgbimage);

    cv::setMouseCallback(RGB_WINDOW, onMouse, 0);

    if(bRenewROI)
    {
        // if (selectRect.width <= 0 || selectRect.height <= 0)
        // {
        //     bRenewROI = false;
        //     //continue;
        // }
        tracker.init(selectRect, rgbimage);
        bBeginKCF = true;
        bRenewROI = false;
        enable_get_depth = false;
    }

    if(bBeginKCF)
    {
        result = tracker.update(rgbimage);
        cv::rectangle(rgbimage, result, cv::Scalar( 0, 255, 255 ), 1, 8 );
        enable_get_depth = true;
    }
    else
        cv::rectangle(rgbimage, selectRect, cv::Scalar(255, 0, 0), 2, 8, 0);

    cv::imshow(RGB_WINDOW, rgbimage);
    cv::waitKey(1);
  }

  void depthCb(const sensor_msgs::ImageConstPtr& msg)
  {
  	cv_bridge::CvImagePtr cv_ptr;
  	try
  	{
  		cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::TYPE_32FC1);
  		cv_ptr->image.copyTo(depthimage);
  	}
  	catch (cv_bridge::Exception& e)
  	{
  		ROS_ERROR("Could not convert from '%s' to 'TYPE_32FC1'.", msg->encoding.c_str());
  	}

    float y_tmp[3];
   
    if(enable_get_depth)
    {
        dist_val[0] = depthimage.at<float>(result.y+result.height/3 , result.x+result.width/3)/scale ;  // 1/3 1/3
        dist_val[1] = depthimage.at<float>(result.y+result.height/3 , result.x+2*result.width/3) /scale; // 1/3 2/3
        dist_val[2] = depthimage.at<float>(result.y+2*result.height/3 , result.x+result.width/3) /scale; // 2/3 1/3
        dist_val[3] = depthimage.at<float>(result.y+2*result.height/3 , result.x+2*result.width/3) /scale; // 2/3 2/3
        dist_val[4] = depthimage.at<float>(result.y+result.height/2 , result.x+result.width/2) /scale; // 1/2 1/2

        ROS_INFO("the distance is %f   ",dist_val[0]);
        ROS_INFO("the distance is %f   ",dist_val[1]);
        ROS_INFO("the distance is %f   ",dist_val[2]);
        ROS_INFO("the distance is %f   ",dist_val[3]);
        ROS_INFO("the distance is %f   ",dist_val[4]);

        y_tmp[0] = (result.x+result.width/3 - cx)*(dist_val[0]+dist_val[2])/(2*fx);   
        y_tmp[1] = (result.x+2*result.width/3 - cx)*(dist_val[1]+dist_val[3])/(2*fx);
        y_tmp[2] = (result.x+result.width/2 - cx)*(dist_val[4])/(fx);


      float distance = 0;
      int num_depth_points = 5;
      for(int i = 0; i < 5; i++)
      {
        if(dist_val[i] > 0.0 && dist_val[i] < 10.0)
          distance += dist_val[i];
        else
          num_depth_points--;
      }

      ROS_INFO("%f",distance);

      if(num_depth_points==0)
        distance = 0;
      else
        distance /= num_depth_points;

      dist_x = distance;
      dist_y = (y_tmp[0]+y_tmp[1]+y_tmp[2])/(3*scale);
      
      ROS_INFO("the distance is %f   ",distance);
      ROS_INFO("the x is %f   ",dist_x);
      ROS_INFO("the y is %f   ",dist_y);
    }

  	//cv::imshow(DEPTH_WINDOW, depthimage);
  	cv::waitKey(1);
  }
};

int main(int argc, char** argv)
{
	ros::init(argc, argv, "kcf_tracker");
  std::string color_image,depth_image;
  ros::param::param<std::string>("~color_image",color_image,"/camera/color/image");
  ros::param::param<std::string>("~depth_image",depth_image,"/camera/depth/image");
	ImageConverter ic(color_image,depth_image);
  ROS_INFO("%s\n",color_image.c_str());
  ROS_INFO("%s\n",depth_image.c_str());
  
	while(ros::ok())
	{
		ros::spinOnce();
    track_pkg::Target target;
    target.x = dist_x;
    target.y = dist_y;

    ROS_INFO("%f",dist_x);
    ROS_INFO("%f",dist_y);

    ic.pub.publish(target);

		if (cvWaitKey(33) == 'q')
      break;
	}

	return 0;
}

