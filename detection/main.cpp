#include <iostream>
#include "CascadeCNN.h"
using namespace std;
using namespace cv;

int main() {
    //vector<string> model_file, trained_file;
    string mean_file = "/home/xileli/Documents/library/caffe/data/ilsvrc12/imagenet_mean.binaryproto";

    vector<string> model_file = {
            "/home/xileli/Documents/program/CNN_face_detection_models/face_12c/face12c_full_conv.prototxt",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_12_cal/deploy.prototxt",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_24c/deploy.prototxt",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_24_cal/deploy.prototxt",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_48c/deploy.prototxt",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_48_cal/deploy.prototxt"
    };

    vector<string> trained_file = {
            "/home/xileli/Documents/program/CNN_face_detection_models/face_12c/face12c_full_conv.caffemodel",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_12_cal/face_12_cal_train_iter_400000.caffemodel",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_24c/face_24c_train_iter_400000.caffemodel",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_24_cal/face_24_cal_train_iter_400000.caffemodel",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_48c/face_48c_train_iter_200000.caffemodel",
            "/home/xileli/Documents/program/CNN_face_detection_models/face_48_cal/face_48_cal_train_iter_300000.caffemodel"
    };


    vector<Rect> rectangles;
    //string img_path = "/home/xileli/Documents/dateset/AFW/WIDER_train/images/4--Dancing/4_Dancing_Dancing_4_33.jpg";
    //string img_path = "/home/xileli/Documents/dateset/AFW/WIDER_train/images/17--Ceremony/17_Ceremony_Ceremony_17_88.jpg";
    string img_path = "/home/xileli/Documents/dateset/megaface/data/daniel/FlickrFinal2/901/9019583@N08/8032652775_0.jpg";
    //string img_path = "/home/xileli/Documents/dateset/megaface/data/daniel/FlickrFinal2/121/12105541@N05/4811548600_1.jpg";
    //string img_path = "/home/xileli/Documents/dateset/megaface/data/daniel/FlickrFinal2/908/9080049@N02/8963775598_2.jpg";
    //string img_path = "/home/xileli/Documents/dateset/AFW/WIDER_train/images/53--Raid/53_Raid_policeraid_53_26.jpg";
    //string img_path = "/home/xileli/Documents/dateset/AFW/WIDER_train/images/53--Raid/53_Raid_policeraid_53_35.jpg";
    //string img_path = "/home/xileli/Documents/dateset/AFW/WIDER_train/images/55--Sports_Coach_Trainer/55_Sports_Coach_Trainer_sportcoaching_55_185.jpg";
    Mat img = imread(img_path);


    CascadeCNN cascadeCNN(model_file,trained_file,mean_file);


    cascadeCNN.detection_test(img, rectangles);
//    cascadeCNN.timer_begin();
//    cascadeCNN.detection(img, rectangles);
//    cascadeCNN.timer_end();

    for(int i = 0; i < rectangles.size(); i++)
        rectangle(img, rectangles[i], Scalar(255, 0, 0));
    imshow("face", img);
    waitKey(0);

    return 0;
}

//int main(int, char**)
//{
//    string mean_file = "/home/xileli/Documents/library/caffe/data/ilsvrc12/imagenet_mean.binaryproto";
//
//    vector<string> model_file = {
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_12c/face12c_full_conv.prototxt",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_12_cal/deploy.prototxt",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_24c/deploy.prototxt",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_24_cal/deploy.prototxt",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_48c/deploy.prototxt",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_48_cal/deploy.prototxt"
//    };
//
//    vector<string> trained_file = {
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_12c/face12c_full_conv.caffemodel",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_12_cal/face_12_cal_train_iter_400000.caffemodel",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_24c/face_24c_train_iter_400000.caffemodel",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_24_cal/face_24_cal_train_iter_400000.caffemodel",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_48c/face_48c_train_iter_200000.caffemodel",
//            "/home/xileli/Documents/program/CNN_face_detection_models/face_48_cal/face_48_cal_train_iter_300000.caffemodel"
//    };
//
//    CascadeCNN cascadeCNN(model_file,trained_file,mean_file);
//
//    VideoCapture cap(0); // open the default camera
//    if(!cap.isOpened())  // check if we succeeded
//        return -1;
//    for(;;)
//    {
//        Mat img;
//        cap >> img; // get a new frame from camera
//
//        vector<Rect> rectangles;
//
//        cascadeCNN.timer_begin();
//        cascadeCNN.detection(img, rectangles);
//        cascadeCNN.timer_end();
//
//        for(int i = 0; i < rectangles.size(); i++)
//            rectangle(img, rectangles[i], Scalar(255, 0, 0));
//        imshow("face", img);
//
//        if(waitKey(30) >= 0) break;
//    }
//    // the camera will be deinitialized automatically in VideoCapture destructor
//    return 0;
//}


