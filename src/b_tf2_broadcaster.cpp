/** ----------------------------------------------------------------------------
 * @file:     b_tf2_broadcaster.cpp
 * @date:     October 30, 2023
 * @datemod:  October 30, 2023
 * @author:   Alejandro Gonzalez-Garcia
 * @email:    alexglzg97@gmail.com
 * 
 * @brief: TF for the bicycle dynamics. 
 * ---------------------------------------------------------------------------*/

#include <math.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/PoseStamped.h>
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"

class TfAndPath{
public:
  nav_msgs::Path path;
  geometry_msgs::PoseStamped pose;
  TfAndPath(){
  path_pub = node.advertise<nav_msgs::Path>("/b/path", 1);
  sub = node.subscribe("/b/odom", 1, &TfAndPath::odomCallback, this);
  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg){
  //void odomCallback(const nav_msgs::Odometry::ConstPtr& msg){
    static tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped transformStamped;
    
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "world";
    transformStamped.child_frame_id = "bicycle";
    transformStamped.transform.translation.x = msg->pose.pose.position.x;
    transformStamped.transform.translation.y = msg->pose.pose.position.y;
    transformStamped.transform.translation.z = 0.0;

    transformStamped.transform.rotation.x = msg->pose.pose.orientation.x;
    transformStamped.transform.rotation.y = msg->pose.pose.orientation.y;
    transformStamped.transform.rotation.z = msg->pose.pose.orientation.z;
    transformStamped.transform.rotation.w = msg->pose.pose.orientation.w;

    br.sendTransform(transformStamped);

    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "world";
    pose.pose.position.x = msg->pose.pose.position.x;
    pose.pose.position.y = msg->pose.pose.position.y;;
    pose.pose.position.z = 0.0;

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "world";
    path.poses.push_back(pose);
    
    path_pub.publish(path);
  }

private:
  ros::NodeHandle node;
  ros::Publisher path_pub;
  ros::Subscriber sub;

};

int main(int argc, char** argv){

  ros::init(argc, argv, "b_tf2_broadcaster");
  TfAndPath tfAndPath;

  while (ros::ok()){
    ros::Rate(50).sleep();
    ros::spinOnce();
  }

  return 0;
}