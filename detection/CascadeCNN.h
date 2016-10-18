//
// Created by 刘阳 on 16/10/17.
//

#ifndef CASCADECNN_CASCADECNN_H
#define CASCADECNN_CASCADECNN_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>

class CascadeCNN {

public:
    CascadeCNN();
    CascadeCNN(const std::vector<std::string> model_file, const std::vector<std::string> trained_file, const std::string& mean_file);
    ~CascadeCNN();

    void detection(const cv::Mat& img, std::vector<cv::Rect>* rectangles);

    void SetMean(const string& mean_file);
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img);

    void detect_12c_net();
    void cal_12c_net();
    void detect_24c_net();
    void cal_24c_net();
    void detect_48c_net();
    void cal_48c_net();

    void local_NMS();
    void global_NMS();
    void global_NMS_specify();

    void genereate_init_rectangles();
    void detect_net(int i);
    void calibrate_net(int i);

    cv::Mat img_;
    std::vector<cv::Rect> rectangles_;
    std::vector<float> confidence_;
    shared_ptr<Net<float> > 12c_, 12cal_, 24c_, 24cal_, 48c_, 48cal_;
    std::vector<shared_ptr<Net<float> >> nets_;
    std::vector<cv::Size> input_geometry_;
    int num_channels_;
    std::vector<cv::Mat> mean_;

    float threshold_NMS_ = 0.3;
    float scale_factor_ = 1.414;
    int small_face_size_ = 48;

    int mode_pu_ = 1; //mode decides processing the neural network with cpu (0) or gpu (1)

    struct Face
    {
        cv::Rect rectangle;
        float confidence;
        double scale;
    };
};


#endif //CASCADECNN_CASCADECNN_H
