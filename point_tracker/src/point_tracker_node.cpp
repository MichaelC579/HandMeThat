#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Point.h>
#include <tf/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


//#include "human_tracker/humans.h"
#include "tracker.h"

#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>


#define CAMERA_PIXEL_WIDTH 640
#define CAMERA_PIXEL_HEIGHT 480



class MyPublisher
{
    public:
    MyPublisher(void);
    ros::Publisher top_pub;
    ros::Publisher bot_pub;
    Tracker hand_track;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    geometry_msgs::PointStamped bot_joint;
    geometry_msgs::PointStamped top_joint;
    tf::TransformListener *tf_listener_;
    cv::Mat image;

    void cloud_callback(const sensor_msgs::PointCloud2 &msg);

};

static ros::ServiceClient						*pClient;
static ros::ServiceServer *pService;
static ros::ServiceServer *pService_torso;
static tf::TransformListener 					*pListener;
static MyPublisher mp;

MyPublisher::MyPublisher(void):cloud(new pcl::PointCloud<pcl::PointXYZRGB>)
{
    hand_track = Tracker();
}


void MyPublisher::cloud_callback(const sensor_msgs::PointCloud2 &msg)
{
    float* joints = hand_track.get_joints(image);
    pcl::fromROSMsg(msg, *cloud);
    ros::Time t = ros::Time::now();
    bot_joint.header.stamp = t;
    bot_joint.header.frame_id = "head_rgbd_sensor_rgb_frame";
    top_joint.header.stamp = t;
    top_joint.header.frame_id = "head_rgbd_sensor_rgb_frame";

    if(joints[0] > 0 && joints[1] > 0 && joints[2] > 0 && joints[3] > 0) {

                int point_idx = joints[0]+joints[1]*CAMERA_PIXEL_WIDTH;
                float point_z=0.0;
                float point_x=0.0;
                float point_y=0.0;

                if ((point_idx<0) || (!pcl::isFinite(cloud->points[point_idx]))){
                    return;
                }
                else{

                   if(cloud->points[point_idx].z)
                   {
                       point_x = cloud->points[point_idx].x;
                       point_y = cloud->points[point_idx].y;
                       point_z = cloud->points[point_idx].z;

                   }
                    //ROS_INFO("bodyparpoint x : %d , y: %d , point idx :%d , point_z : %.3f ",bodypart_3d.x, bodypart_3d.y, point_idx, point_z);
                    top_joint.point.x =point_x;
                    top_joint.point.y =point_y;
                    top_joint.point.z =point_z;

                }

                point_idx = joints[2]+joints[3]*CAMERA_PIXEL_WIDTH;
                point_z=0.0;
                point_x=0.0;
                point_y=0.0;

                if ((point_idx<0) || (!pcl::isFinite(cloud->points[point_idx]))){
                    return;
                }
                else{

                   if(cloud->points[point_idx].z)
                   {
                       point_x = cloud->points[point_idx].x;
                       point_y = cloud->points[point_idx].y;
                       point_z = cloud->points[point_idx].z;

                   }
                    //ROS_INFO("bodyparpoint x : %d , y: %d , point idx :%d , point_z : %.3f ",bodypart_3d.x, bodypart_3d.y, point_idx, point_z);
                    bot_joint.point.x =point_x;
                    bot_joint.point.y =point_y;
                    bot_joint.point.z =point_z;

                }

    top_pub.publish(top_joint);
    bot_pub.publish(bot_joint);
}
}



void callback(const sensor_msgs::PointCloud2 &cloud)
{
	mp.cloud_callback(cloud);
}

void callback2(const sensor_msgs::Image::ConstPtr &rgb_image)
{   
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::RGB8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat imgI(rgb_image->height, rgb_image->width, CV_8UC3);
    memcpy(imgI.data, rgb_image->data.data(), rgb_image->data.size());
    mp.image = imgI;
}


int main(int argc, char *argv[])
{
	ros::init(argc, argv, "point_tracker");
	ros::NodeHandle nh;
	ros::start();

	tf::TransformListener listener(ros::Duration(5));
	pListener = &listener;
	pListener->waitForTransform("base_link", "head_rgbd_sensor_rgb_frame", ros::Time::now(), ros::Duration(5.));
    ros::Subscriber subscriber1 = nh.subscribe( "/hsrb/head_rgbd_sensor/rgb/image_rect_color", 1, callback2);
	ros::Subscriber subscriber2 = nh.subscribe( "/hsrb/head_rgbd_sensor/depth_registered/rectified_points", 1, callback);
    mp.top_pub = nh.advertise<geometry_msgs::PointStamped>("/hand_track/top_joint", 0);
    mp.top_pub = nh.advertise<geometry_msgs::PointStamped>("/hand_track/bot_joint", 0);
    mp.tf_listener_ = new tf::TransformListener(nh);

	ROS_INFO("Tracker started");

	ros::spin();

	ros::shutdown();

	return 0;
}
