//
// Created by xileli on 10/31/16.
//

#ifndef FACE_DETECTION_FDDB_DETECTION_H
#define FACE_DETECTION_FDDB_DETECTION_H

#include <string>
#include <vector>
#include <fstream>
#include "../../../detection/CascadeCNN.h"

class fddb_detection {

public:
    fddb_detection();
    fddb_detection(std::string path);
    fddb_detection(std::string dataset_path, CascadeCNN * cascadeCNN);

    void run();

    bool file_list_read();
    bool file_list_read(string dataset_path);
    bool img_path_read(int i);
    bool img_path_read(string path);
    bool img_read(std::string path);
    bool generate_txt();
    bool txt_write(std::vector<cv::Rect> rects, std::vector<float> confidence, std::string img_path);
    bool img_write(std::vector<cv::Rect> rects, std::vector<float> confidence, std::string img_path);
    std::string img_path_convert(std::string path);

    CascadeCNN * cascadeCNN_;
    std::string dataset_path_;
    std::vector<std::string> file_list_;
    std::vector<std::string> img_path_;
    cv::Mat cur_img_;
    int i_;
    int txt_write_state_ = 0;
    int img_write_state_ = 1;

    std::string mean_file = "/home/xileli/Documents/library/caffe/data/ilsvrc12/imagenet_mean.binaryproto";

    std::vector<std::string> model_file = {
            "/home/xileli/Documents/program/CNN_face_detection_models/face_12c/face12c_full_conv.prototxt",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_12_cal/deploy.prototxt",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_24c/deploy.prototxt",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_24_cal/deploy.prototxt",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_48c/deploy.prototxt",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_48_cal/deploy.prototxt"
    };

    std::vector<std::string> trained_file = {
            "/home/xileli/Documents/program/CNN_face_detection_models/face_12c/face12c_full_conv.caffemodel",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_12_cal/face_12_cal_train_iter_400000.caffemodel",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_24c/face_24c_train_iter_400000.caffemodel",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_24_cal/face_24_cal_train_iter_400000.caffemodel",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_48c/face_48c_train_iter_200000.caffemodel",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_48_cal/face_48_cal_train_iter_300000.caffemodel"
    };
};


#endif //FACE_DETECTION_FDDB_DETECTION_H
