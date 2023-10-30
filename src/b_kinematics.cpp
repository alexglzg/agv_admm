/** ----------------------------------------------------------------------------
 * @file:     u_kinematics.cpp
 * @date:     October 30, 2023
 * @datemod:  October 30, 2023
 * @author:   Alejandro Gonzalez-Garcia
 * @email:    alexglzg97@gmail.com
 * 
 * @brief: Kinematics for Bicycle. 
 * ---------------------------------------------------------------------------*/

#include <iostream>
#include "ros/ros.h"
#include "geometry_msgs/Pose2D.h"
#include "geometry_msgs/Vector3.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Float64.h"
#include "std_msgs/UInt8.h"
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <tf2/LinearMath/Quaternion.h>

using namespace Eigen;

class DynamicModel
{
public:
    float integral_step;

    //Identified parameters
    float max_speed;
    float max_steering;
    float L;
    float lr;

    tf2::Quaternion myQuaternion;

    float x;
    float y;
    float psi;

    Vector3f xs;
    Vector3f xs_dot_last;
    Vector3f xs_dot;

    float v;
    float delta;

    geometry_msgs::Pose2D dynamic_pose; //inertial navigation system pose [North East Yaw] or [x y psi]
    nav_msgs::Odometry odom;
    
    DynamicModel()
    {
        //ROS Publishers for each required simulated ins_2d data
        inertial_pose_pub = n.advertise<geometry_msgs::Pose2D>("/b/pose", 1);
        odom_pub = n.advertise<nav_msgs::Odometry>("/b/odom", 1);

        vel_sub = n.subscribe("/b/velocity_reference", 1, &DynamicModel::vel_callback, this);

        static const float starting_pose = 1.0;

        n.param("u_kinematics/x", x, starting_pose);
        n.param("u_kinematics/y", y, starting_pose);
        n.param("u_kinematics/psi", psi, starting_pose);

        max_speed = 1.0;
        max_steering = 1.2;
        L = 0.5;
        lr = L/2;

        v = 0.0,
        delta = 0.0;

        xs << x, y, psi;
        xs_dot_last << 0.0, 0.0, 0.0;
    }

    void vel_callback(const geometry_msgs::Pose2D::ConstPtr& _vel)
    {
        v = _vel->x; //forward force in N
        delta = _vel->theta; //rotational moment in Nm
    }

    void time_step()
    {
        
        xs_dot << v*cos(psi), v*sin(psi), v*tan(delta)/L;
        xs = integral_step * (xs_dot + xs_dot_last)/2 + xs; //integral [x y psi]
        xs_dot_last = xs_dot;

        x = xs(0); //position in x
        y = xs(1); //position in y
        psi = xs(2); //orientation psi
        //Wrap to [-pi pi]
        if (std::abs(psi) > 3.141592){
            psi = (psi/std::abs(psi))*(std::abs(psi)-2*3.141592);
            xs(2) = psi;
        }
        dynamic_pose.x = x;
        dynamic_pose.y = y;
        dynamic_pose.theta = psi;
        odom.pose.pose.position.x = x;
        odom.pose.pose.position.y = -y;
        odom.pose.pose.position.z = 0;
        myQuaternion.setRPY(0.0,0.0,-psi);
        odom.pose.pose.orientation.x = myQuaternion[0];
        odom.pose.pose.orientation.y = myQuaternion[1];
        odom.pose.pose.orientation.z = myQuaternion[2];
        odom.pose.pose.orientation.w = myQuaternion[3];
        odom.twist.twist.linear.x = v;
        odom.twist.twist.linear.y = 0.0;
        odom.twist.twist.linear.z = 0.0;
        odom.twist.twist.angular.x = 0.0;
        odom.twist.twist.angular.y = 0.0;
        odom.twist.twist.angular.z = -xs_dot(2);
        odom.header.stamp = ros::Time::now();
        odom.header.frame_id = "world";
        odom.child_frame_id = "bicycle";

        //Data publishing
        inertial_pose_pub.publish(dynamic_pose);
        odom_pub.publish(odom);
        
    }

private:
    ros::NodeHandle n;

    ros::Publisher inertial_pose_pub;
    ros::Publisher odom_pub;

    ros::Subscriber vel_sub;
};

//Main
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "b_kinematics");
    DynamicModel dynamicModel;
    dynamicModel.integral_step = 0.01;
    int rate = 100;
    ros::Rate loop_rate(rate);

  while (ros::ok())
  {
    dynamicModel.time_step();
    ros::spinOnce();
    loop_rate.sleep();
  }

    return 0;
}
