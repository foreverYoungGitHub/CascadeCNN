//
// Created by Yang on 16/10/17.
//

#ifndef CASCADECNN_CASCADECNN_H
#define CASCADECNN_CASCADECNN_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <fstream>

using namespace caffe;

class CascadeCNN {

public:
    CascadeCNN();
    CascadeCNN(const std::vector<std::string> model_file, const std::vector<std::string> trained_file, const std::string& mean_file);
    ~CascadeCNN();

    void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles);
    void detection(const cv::Mat img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence);
    void detection_test(const cv::Mat img, std::vector<cv::Rect>& rectangles);
    

    std::vector<float> Predict(const cv::Mat& img, int i);
    std::vector<float> Predict(const std::vector<cv::Mat> imgs, int i);

    void SetMean(const std::string& mean_file);
    cv::Mat SetMean(cv::Mat img, const std::string& mean_file);
    void WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat>* input_channels, int i);
    void WrapBatchInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i);
    void WrapInputLayer(const cv::Mat& img, int i);
    void Preprocess(const cv::Mat& img);

    void detect_12c_net();
    void cal_12c_net();
    void detect_24c_net();
    void cal_24c_net();
    void detect_48c_net();
    void cal_48c_net();

    void local_NMS();
    void local_NMS_test();
    void global_NMS();
    void global_NMS_specify();

    float IoU(cv::Rect rect1, cv::Rect rect2);
    float IoM(cv::Rect rect1, cv::Rect rect2);

    void generate_init_rectangles();
    void detect_net(int i);
    void detect_net_batch(int i);
    void calibrate_net(int i);
    void calibrate_net_batch(int i);
    void calibrate(std::vector<float> prediction, int j);


    void img_show(cv::Mat img, std::string name);
    void timer_begin();
    void timer_end();
    void record(double num);

    cv::Mat crop(cv::Mat img, cv::Rect rect);
    void clear();
    cv::Mat img_;
    std::vector<cv::Rect> rectangles_;
    std::vector<float> confidence_;

    std::vector<std::shared_ptr<Net<float>>> nets_;
    std::vector<cv::Size> input_geometry_;
    int num_channels_;
    std::vector<cv::Mat> mean_;
    std::string mean_file_;

    int strip_ = 4;
    float threshold_confidence_ = 0.05;
    float threshold_NMS_ = 0.3;
    float scale_factor_ = 1.414;
    int dimension_ = 48;
    int mode_pu_ = 0; //mode decides processing the neural network with cpu (0) or gpu (1)

    std::chrono::high_resolution_clock::time_point time_begin_, time_end_;
};


#endif //CASCADECNN_CASCADECNN_H
