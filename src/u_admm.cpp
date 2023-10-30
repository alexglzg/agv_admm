/** ----------------------------------------------------------------------------
 * @file:     u_admm.cpp
 * @date:     October 27, 2023
 * @datemod:  October 27, 2023
 * @author:   Alejandro Gonzalez-Garcia
 * @email:    alexglzg97@gmail.com
 * 
 * @brief: ADMM algorithm for distributed heterogeneous AGV control. 
 * ---------------------------------------------------------------------------*/

#include <iostream>
#include "ros/ros.h"
#include "geometry_msgs/Pose2D.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Vector3.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "std_msgs/Float64.h"
#include "std_msgs/UInt16.h"
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <tf2/LinearMath/Quaternion.h>
#include "agv_admm/FloatArray.h"
#include <casadi/casadi.hpp>

using namespace Eigen;
using namespace casadi;

class UAdmm
{
public:

    Vector2d reference;
    double x;
    double y;
    double psi;
        
    bool admm_initialized;
    int counter;
    int counter_mpc;
    bool ocpX_flag;
    bool ocpZ_flag;
    bool ref_flag;

    int Nhor;
    int Nhor_plus_one;
    int ocpX_states;
    double mu;
    
    std::string FUNCTIONS_DIR;
    std::string FUNCTIONS_DIR_DEFAULT;
    casadi::Function ocpX_function;
    casadi::Function ocpZ_function;
    //casadi::Function ocpC_function;

    agv_admm::FloatArray trajectory;
    agv_admm::FloatArray local_copies;
    agv_admm::FloatArray lambda_multipliers;
    geometry_msgs::Pose2D vel_ref;
    std_msgs::Float64 ocpX_time;
    std_msgs::Float64 ocpZ_time;
    //std_msgs::Float64 cbf_time;
    //nav_msgs::Path horizon_path;
    //geometry_msgs::PoseStamped horizon_pose;

    // ocpX Parameters
    MatrixXd xref;
    MatrixXd yref;
    MatrixXd multi_i;
    MatrixXd copy_i;
    MatrixXd multi_ji;
    MatrixXd copy_ji;
    MatrixXd X_0;

    // ocpZ Parameters
    MatrixXd z_x;
    MatrixXd z_y;
    MatrixXd copy_ij;
    //multi_i is used again
    MatrixXd traj_i;
    MatrixXd multi_ij;
    MatrixXd traj_ij;
    MatrixXd diam;

    /*// CBF Parameters
    // Reusing MatrixXd diam;
    MatrixXd x_i;
    MatrixXd y_i;
    MatrixXd u_mpc;
    MatrixXd v_mpc;
    MatrixXd C_j;
    MatrixXd V_j;*/

    // ocpX Outputs
    MatrixXd x_res;
    MatrixXd y_res;
    MatrixXd v_res;
    MatrixXd w_res;

    // ocpZ Outputs
    MatrixXd zx_res;
    MatrixXd zy_res;
    MatrixXd copy_ij_res;

    /*// CBF Outputs
    MatrixXd u_cbf;
    MatrixXd v_cbf;*/

    // ocpX vectors
    std::vector<const double*> arg_ocpX;
    std::vector<double*> res_ocpX;
    std::vector<casadi_int> iw_ocpX;
    std::vector<double> w_ocpX;
    int mem_ocpX;

    // ocpZ vectors
    std::vector<const double*> arg_ocpZ;
    std::vector<double*> res_ocpZ;
    std::vector<casadi_int> iw_ocpZ;
    std::vector<double> w_ocpZ;
    int mem_ocpZ;

    /*// CBF vectors
    std::vector<const double*> arg_ocpC;
    std::vector<double*> res_ocpC;
    std::vector<casadi_int> iw_ocpC;
    std::vector<double> w_ocpC;
    int mem_ocpC;*/

    std::vector<double> new_trajectory;
    std::vector<double> new_local_copies;
    std::vector<double> new_lambda_multipliers;

    std::vector<double> neighbor_trajectory;
    std::vector<double> neighbor_local_copies;
    std::vector<double> neighbor_lambda_multipliers;

    
    UAdmm()
    {
        //ROS Publishers and Subscribers
        trajectory_pub = nh.advertise<agv_admm::FloatArray>("/u/trajectory", 1);
        local_copies_pub = nh.advertise<agv_admm::FloatArray>("/u/local_copies", 1);
        lambda_multipliers_pub = nh.advertise<agv_admm::FloatArray>("/u/lambda_multipliers", 1);
        vel_ref_pub = nh.advertise<geometry_msgs::Pose2D>("/u/velocity_reference", 1);
        path_pub = nh.advertise<nav_msgs::Path>("/u/horizon_path", 1);
        ocpX_time_pub = nh.advertise<std_msgs::Float64>("/u/ocpX_time", 1);
        ocpZ_time_pub = nh.advertise<std_msgs::Float64>("/u/ocpZ_time", 1);
        //cbf_time_pub = nh.advertise<std_msgs::Float64>("cbf_time", 1);

        guessedi_path_pub = nh.advertise<nav_msgs::Path>("/u/guessedi_horizon_path", 1);
        guessedj_path_pub = nh.advertise<nav_msgs::Path>("/u/guessedj_horizon_path", 1);

        reference_pose_sub = nh.subscribe("/u/reference_pose", 1, &UAdmm::reference_callback, this);
        counter_reset_sub = nh.subscribe("/counter_restart", 1, &UAdmm::reset_callback, this);
        pose_sub = nh.subscribe("/u/pose", 1, &UAdmm::pose_callback, this);

        trajectory_sub = nh.subscribe("/b/trajectory", 1, &UAdmm::trajectory_callback, this);
        local_copies_sub = nh.subscribe("/b/local_copies", 1, &UAdmm::local_copies_callback, this);
        lambda_multipliers_sub = nh.subscribe("/b/lambda_multipliers", 1, &UAdmm::lambda_multipliers_callback, this);

        FUNCTIONS_DIR_DEFAULT = "/home/alex/agv_ws/src/agv_admm/scripts/ipopt_definitions";
        nh.param("swarm/Nhor", Nhor, 40);
        nh.param("swarm/ocpX_states", ocpX_states, 2);
        nh.param("swarm/mu", mu, 10.0);
        nh.param("functions/path", FUNCTIONS_DIR, FUNCTIONS_DIR_DEFAULT);
        Nhor_plus_one = Nhor + 1;

        ocpX_function = casadi::Function::load(FUNCTIONS_DIR + "/u_ocpX.casadi");
        ocpZ_function = casadi::Function::load(FUNCTIONS_DIR + "/u_ocpZ.casadi");
        //ocpC_function = casadi::Function::load(FUNCTIONS_DIR + "/ocpCBF.casadi");

        // initialize ocpX parameters
        xref = MatrixXd::Zero(1,1);
        yref = MatrixXd::Zero(1,1);
        multi_i = MatrixXd::Zero(ocpX_states, Nhor_plus_one);
        copy_i = MatrixXd::Zero(ocpX_states, Nhor_plus_one);
        multi_ji = MatrixXd::Zero(ocpX_states, Nhor_plus_one);
        copy_ji = MatrixXd::Zero(ocpX_states, Nhor_plus_one);
        X_0 = MatrixXd::Zero(ocpX_states+1, 1);

        // initialize ocpZ parameters
        z_x = MatrixXd::Zero(1,Nhor_plus_one);
        z_y = MatrixXd::Zero(1,Nhor_plus_one);
        copy_ij = MatrixXd::Zero(ocpX_states, Nhor_plus_one);
        // multi_i is already initialized in ocpX
        traj_i = MatrixXd::Zero(ocpX_states, Nhor_plus_one);
        multi_ij = MatrixXd::Zero(ocpX_states, Nhor_plus_one);
        traj_ij = MatrixXd::Zero(ocpX_states, Nhor_plus_one);
        diam = MatrixXd::Zero(1,1);
        diam(0,0) = 0.4;
        
        /*// initialize CBF parameters
        // diam is already initialized in ocpZ
        x_i = MatrixXd::Zero(1,1);
        y_i = MatrixXd::Zero(1,1);
        u_mpc = MatrixXd::Zero(1, 1);
        v_mpc = MatrixXd::Zero(1, 1);
        C_j = MatrixXd::Zero(ocpX_states, 1);
        V_j = MatrixXd::Zero(ocpX_states, 1);*/
        
        // initialize ocpX outputs
        x_res = MatrixXd::Zero(1, Nhor_plus_one);
        y_res = MatrixXd::Zero(1, Nhor_plus_one);
        v_res = MatrixXd::Zero(1, Nhor);
        w_res = MatrixXd::Zero(1, Nhor);

        // initialize ocpZ outputs
        zx_res = MatrixXd::Zero(1, Nhor_plus_one);
        zy_res = MatrixXd::Zero(1, Nhor_plus_one);
        copy_ij_res = MatrixXd::Zero(ocpX_states, Nhor_plus_one);

        /*// initialize CBF outputs
        u_cbf = MatrixXd::Zero(1, 1);
        v_cbf = MatrixXd::Zero(1, 1);*/

        // initialize ocpX vectors
        arg_ocpX = std::vector<const double*>(ocpX_function.sz_arg());
        res_ocpX = std::vector<double*>(ocpX_function.sz_res());
        iw_ocpX = std::vector<casadi_int>(ocpX_function.sz_iw());
        w_ocpX = std::vector<double>(ocpX_function.sz_w());
        mem_ocpX = ocpX_function.checkout();

        //initialize ocpZ vectors
        arg_ocpZ = std::vector<const double*>(ocpZ_function.sz_arg());
        res_ocpZ = std::vector<double*>(ocpZ_function.sz_res());
        iw_ocpZ = std::vector<casadi_int>(ocpZ_function.sz_iw());
        w_ocpZ = std::vector<double>(ocpZ_function.sz_w());
        mem_ocpZ = ocpZ_function.checkout();

        /*//initialize CBF vectors
        arg_ocpC = std::vector<const double*>(ocpC_function.sz_arg());
        res_ocpC = std::vector<double*>(ocpC_function.sz_res());
        iw_ocpC = std::vector<casadi_int>(ocpC_function.sz_iw());
        w_ocpC = std::vector<double>(ocpC_function.sz_w());
        mem_ocpC = ocpC_function.checkout();*/

        // initialize ocpX args
        arg_ocpX[0] = &xref(0,0);
        arg_ocpX[1] = &yref(0,0);
        arg_ocpX[2] = &multi_i(0,0);
        arg_ocpX[3] = &copy_i(0,0);
        arg_ocpX[4] = &multi_ji(0,0);
        arg_ocpX[5] = &copy_ji(0,0);
        arg_ocpX[6] = &X_0(0,0);

        // initialize ocpZ args
        arg_ocpZ[0] = &z_x(0,0);
        arg_ocpZ[1] = &z_y(0,0);
        arg_ocpZ[2] = &copy_ij(0,0);
        arg_ocpZ[3] = &multi_i(0,0);
        arg_ocpZ[4] = &traj_i(0,0);
        arg_ocpZ[5] = &multi_ij(0,0);
        arg_ocpZ[6] = &traj_ij(0,0);
        arg_ocpZ[7] = &diam(0,0);

        /*// initialize ocpC args
        arg_ocpC[0] = &diam(0,0);
        arg_ocpC[1] = &x_i(0,0);
        arg_ocpC[2] = &y_i(0,0);
        arg_ocpC[3] = &u_mpc(0,0);
        arg_ocpC[4] = &v_mpc(0,0);
        arg_ocpC[5] = &C_j(0,0);
        //arg_ocpC[6] = &V_j(0,0);*/

        // initialize ocpX results
        res_ocpX[0] = &x_res(0,0);
        res_ocpX[1] = &y_res(0,0);
        res_ocpX[2] = &v_res(0,0);
        res_ocpX[3] = &w_res(0,0);

        // initialize ocpZ results
        res_ocpZ[0] = &zx_res(0,0);
        res_ocpZ[1] = &zy_res(0,0);
        res_ocpZ[2] = &copy_ij_res(0,0);

        /*// initialize CBF results
        res_ocpC[0] = &u_cbf(0,0);
        res_ocpC[1] = &v_cbf(0,0);*/

        admm_initialized = false;
        counter = 0;
        counter_mpc = 0;
        ocpX_flag = true;
        ocpZ_flag = false;

        reference(0) = 2.0;
        reference(1) = 0.0;

        x = 0.0;
        y = 0.0;
        psi = 0.0;

        neighbor_trajectory.resize(ocpX_states*Nhor_plus_one);
        neighbor_local_copies.resize(ocpX_states*Nhor_plus_one);
        neighbor_lambda_multipliers.resize(ocpX_states*Nhor_plus_one);

    }

    void pose_callback(const geometry_msgs::Pose2D::ConstPtr& _pose)
    {
        x = _pose->x; //ref in x
        y = _pose->y; //ref in y
        psi = _pose->theta;
    }

    void reference_callback(const geometry_msgs::Pose2D::ConstPtr& _ref)
    {
        reference(0) = _ref->x; //ref in x
        reference(1) = _ref->y; //ref in y
        diam(0,0) = _ref->theta;
        ref_flag = true;
    }

    void reset_callback(const std_msgs::UInt16::ConstPtr& _counter)
    {
        counter = _counter->data;
        counter_mpc = _counter->data;
        ROS_ERROR("Counter reset");
    }

    void trajectory_callback(const agv_admm::FloatArray::ConstPtr& _traj)
    {
        neighbor_trajectory = _traj->data;
    }

    void local_copies_callback(const agv_admm::FloatArray::ConstPtr& _lc)
    {
        neighbor_local_copies = _lc->data;
    }

    void lambda_multipliers_callback(const agv_admm::FloatArray::ConstPtr& _lm)
    {
        neighbor_lambda_multipliers = _lm->data;
    }

    void time_step()
    {
        if (!ref_flag){
            return;
        }

        if (ocpX_flag){
            // update ocpX parameters (reference and initial state)
            xref(0,0) = reference(0);
            yref(0,0) = reference(1);
            X_0(0,0) = x;
            X_0(1,0) = y;
            X_0(2,0) = psi;
            
            // solve ocpX
            ocpX_function(casadi::get_ptr(arg_ocpX), casadi::get_ptr(res_ocpX), casadi::get_ptr(iw_ocpX), casadi::get_ptr(w_ocpX), mem_ocpX);

            // unpackage solved trajectory for communication with agents (x_i, y_i) and ocpZ parameter update (initial z_x, z_y, trajectory x_i,y_i)
            new_trajectory.clear();
            for (int i = 0; i < Nhor_plus_one; i++)
            {
                new_trajectory.push_back(x_res(0,i));
                z_x(0,i) = x_res(0,i);
                traj_i(0,i) = x_res(0,i);
            } 
            for (int i = 0; i < Nhor_plus_one; i++)
            {
                new_trajectory.push_back(y_res(0,i));
                z_y(0,i) = y_res(0,i);
                traj_i(1,i) = y_res(0,i);
            }
            if (admm_initialized == false){
                new_trajectory.clear();
                for (int i = 0; i < Nhor_plus_one; i++)
                {
                    new_trajectory.push_back(x);
                    z_x(0,i) = x;
                    traj_i(0,i) = x;
                } 
                for (int i = 0; i < Nhor_plus_one; i++)
                {
                    new_trajectory.push_back(y);
                    z_y(0,i) = y;
                    traj_i(1,i) = y;
                }
            }
            trajectory.data = new_trajectory;
            trajectory_pub.publish(trajectory);

            ocpX_time.data = ocpX_function.stats().at("t_wall_total");
            ocpX_time_pub.publish(ocpX_time);

            nav_msgs::Path horizon_path;
            geometry_msgs::PoseStamped horizon_pose;
            for (int i = 0; i < Nhor_plus_one; i++){
                horizon_pose.header.stamp = ros::Time::now();
                horizon_pose.header.frame_id = "world";
                horizon_pose.pose.position.x = traj_i(0,i);
                horizon_pose.pose.position.y = -traj_i(1,i);
                horizon_pose.pose.position.z = 0.0;

                horizon_path.header.stamp = ros::Time::now();
                horizon_path.header.frame_id = "world";
                horizon_path.poses.push_back(horizon_pose);
            }
            path_pub.publish(horizon_path);

        }        

        if (ocpZ_flag){

            // receive trajectories of other agent for ocpZ parameter (x_ij, y_ij)
            for (int l = 0; l < Nhor_plus_one; l++) {
                traj_ij(0,l) = neighbor_trajectory[l];
                traj_ij(1,l) = neighbor_trajectory[l+Nhor_plus_one];
                copy_ij(0,l) = neighbor_trajectory[l];
                copy_ij(1,l) = neighbor_trajectory[l+Nhor_plus_one];
            }

            if (admm_initialized == false){
                for (int l = 0; l < Nhor_plus_one; l++) {
                    traj_ij(0,l) = x;
                    traj_ij(1,l) = y;
                    copy_ij(0,l) = x;
                    copy_ij(1,l) = y;
                }
            }

            // solve ocpZ
            ocpZ_function(casadi::get_ptr(arg_ocpZ), casadi::get_ptr(res_ocpZ), casadi::get_ptr(iw_ocpZ), casadi::get_ptr(w_ocpZ), mem_ocpZ);

            // unpackage new trajectory estimates/local copies (Z_ij) for communication with other agents
            new_local_copies.clear();
            for (int k = 0; k < ocpX_states; k++){
                for (int l = 0; l < Nhor_plus_one; l++) {
                        new_local_copies.push_back(copy_ij_res(k,l));
                }
            } 
            if (admm_initialized == false){
                new_local_copies.clear();
                for (int i = 0; i < ((ocpX_states)*(Nhor_plus_one)); i++)
                {
                    new_local_copies.push_back(y);
                }
            }
            local_copies.data = new_local_copies;
            local_copies_pub.publish(local_copies);

            // update lambda multipliers i and ij
            new_lambda_multipliers.clear();        
            for (int i = 0; i < Nhor_plus_one; i++){
                multi_i(0,i) = multi_i(0,i) + mu*(zx_res(0,i)-x_res(0,i)); 
                multi_i(1,i) = multi_i(1,i) + mu*(zy_res(0,i)-y_res(0,i)); 
            }
            for (int k = 0; k < ocpX_states; k++){
                for (int l = 0; l < Nhor_plus_one; l++) {
                    multi_ij(k,l) = multi_ij(k,l) + mu*(copy_ij_res(k,l) - traj_ij(k,l));
                }
            }
            if (admm_initialized == false){
                for (int i = 0; i < Nhor_plus_one; i++){
                    multi_i(0,i) = 0.0; 
                    multi_i(1,i) = 0.0; 
                }
                for (int k = 0; k < ocpX_states; k++){
                    for (int l = 0; l < Nhor_plus_one; l++) {
                        multi_ij(k,l) = 0.0;
                    }
                }
                admm_initialized = true;
            }
            for (int k = 0; k < ocpX_states; k++){
                for (int l = 0; l < Nhor_plus_one; l++) {
                    new_lambda_multipliers.push_back(multi_ij(k,l));
                }
            }
            lambda_multipliers.data = new_lambda_multipliers;
            lambda_multipliers_pub.publish(lambda_multipliers);

            ocpZ_time.data = ocpZ_function.stats().at("t_wall_total");
            ocpZ_time_pub.publish(ocpZ_time);

        }

        // update new trajectory estimate/local copy (Z_i) for ocpX
        for (int i = 0; i < Nhor_plus_one; i++)
        {
            copy_i(0,i) = zx_res(0,i);
        } 
        for (int i = 0; i < Nhor_plus_one; i++)
        {
            copy_i(1,i) = zy_res(0,i);
        }

        // update other agents' guesses of current agent trajectory (Z_ji) ocpX parameter
        for (int l = 0; l < Nhor_plus_one; l++) {
            copy_ji(0,l) = neighbor_local_copies[l];
            copy_ji(1,l) = neighbor_local_copies[l+Nhor_plus_one];
        }

        nav_msgs::Path horizon_path;
        geometry_msgs::PoseStamped horizon_pose;
        //horizon_path.poses.clear();
        for (int i = 0; i < Nhor_plus_one; i++){
            horizon_pose.header.stamp = ros::Time::now();
            horizon_pose.header.frame_id = "world";
            horizon_pose.pose.position.x = copy_i(0,i);
            horizon_pose.pose.position.y = -copy_i(1,i);
            horizon_pose.pose.position.z = 0.0;

            horizon_path.header.stamp = ros::Time::now();
            horizon_path.header.frame_id = "world";
            horizon_path.poses.push_back(horizon_pose);
        }
        guessedi_path_pub.publish(horizon_path);

        //nav_msgs::Path horizon_path;
        //geometry_msgs::PoseStamped horizon_pose;
        horizon_path.poses.clear();
        for (int i = 0; i < Nhor_plus_one; i++){
            horizon_pose.header.stamp = ros::Time::now();
            horizon_pose.header.frame_id = "world";
            horizon_pose.pose.position.x = copy_ji(0,i);
            horizon_pose.pose.position.y = -copy_ji(1,i);
            horizon_pose.pose.position.z = 0.0;

            horizon_path.header.stamp = ros::Time::now();
            horizon_path.header.frame_id = "world";
            horizon_path.poses.push_back(horizon_pose);
        }
        guessedj_path_pub.publish(horizon_path);

        // update other agents' multipliers (lambda_ji) ocpX parameter
        for (int l = 0; l < Nhor_plus_one; l++) {
            multi_ji(0,l) = neighbor_lambda_multipliers[l];
            multi_ji(1,l) = neighbor_lambda_multipliers[l+Nhor_plus_one];
        }

        if (ocpX_flag){
            ocpX_flag = false;
            ocpZ_flag = true;
        }
        else{
            ocpX_flag = true;
            ocpZ_flag = false;
        }

        if (counter > 400){
            if (counter_mpc > 1){
                vel_ref.x = v_res(0,0);
                vel_ref.y = 0.0;
                vel_ref.theta = w_res(0,0);
                vel_ref_pub.publish(vel_ref);
                counter_mpc = 0;
            }
            counter_mpc += 1;
        }
        else{
            if (counter_mpc > 1){
                vel_ref.x = 0.0;
                vel_ref.y = 0.0;
                vel_ref.theta = 0.0;
                vel_ref_pub.publish(vel_ref);
                counter_mpc = 0;
            }
            counter_mpc += 1;
        }
        counter += 1;

        if (counter == 400){
            ROS_ERROR("ADMM initialized");
            ROS_ERROR("uni x_ref %f, y_ref %f", xref(0,0), yref(0,0));
        }

    }

private:
    ros::NodeHandle nh;
    ros::Publisher trajectory_pub;
    ros::Publisher local_copies_pub;
    ros::Publisher lambda_multipliers_pub;
    ros::Publisher vel_ref_pub;
    ros::Publisher path_pub;
    ros::Publisher ocpX_time_pub;
    ros::Publisher ocpZ_time_pub;
    //ros::Publisher cbf_time_pub;
    ros::Subscriber pose_sub;
    ros::Subscriber reference_pose_sub;
    ros::Subscriber counter_reset_sub;
    ros::Subscriber trajectory_sub;
    ros::Subscriber local_copies_sub;
    ros::Subscriber lambda_multipliers_sub;

    ros::Publisher guessedi_path_pub;
    ros::Publisher guessedj_path_pub;

};

//Main
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "u_admm");
    UAdmm uAdmm;
    int rate = 40;
    ros::Rate loop_rate(rate);
    ros::Duration(2).sleep();

    while (ros::ok())
    {
        uAdmm.time_step();
        ros::spinOnce();
        loop_rate.sleep();
    }
    uAdmm.ocpX_function.release(uAdmm.mem_ocpX);
    uAdmm.ocpZ_function.release(uAdmm.mem_ocpZ);
    //uAdmm.ocpC_function.release(uAdmm.mem_ocpC);

    return 0;
}