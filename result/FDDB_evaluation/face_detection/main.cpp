#include <iostream>
#include "fddb_detection.h"
#include "ellipse_transform.h"
//int main() {
//
//    std::string mean_file = "/home/xileli/Documents/library/caffe/data/ilsvrc12/imagenet_mean.binaryproto";
//
//    std::vector<std::string> model_file = {
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_12c/face12c_full_conv.prototxt",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_12_cal/deploy.prototxt",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_24c/deploy.prototxt",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_24_cal/deploy.prototxt",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_48c/deploy.prototxt",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_48_cal/deploy.prototxt"
//    };
//
//    std::vector<std::string> trained_file = {
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_12c/face12c_full_conv.caffemodel",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_12_cal/face_12_cal_train_iter_400000.caffemodel",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_24c/face_24c_train_iter_400000.caffemodel",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_24_cal/face_24_cal_train_iter_400000.caffemodel",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_48c/face_48c_train_iter_200000.caffemodel",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_48_cal/face_48_cal_train_iter_300000.caffemodel"
//    };
//
//    CascadeCNN * cascadeCNN;
//    cascadeCNN = new CascadeCNN(model_file, trained_file, mean_file);
//
//    string dataset_path = "/home/xileli/Documents/dateset/FDDB/";
//
//    fddb_detection detection(dataset_path, cascadeCNN);
//
//    detection.run();
//}

int main()
{
    vector<string> path = {
            "/home/xileli/Documents/program/CascadeCNN/result/FDDB_evaluation/FDDB-folds/fileList.txt",
            "/home/xileli/Documents/program/CascadeCNN/result/FDDB_evaluation/FDDB-folds/ellipseList.txt",
            "/home/xileli/Documents/program/CascadeCNN/result/FDDB_evaluation/FDDB-folds/CascadeCNN.txt"
    };
    ellipse_transform ellipse;
    ellipse.run(path);
}