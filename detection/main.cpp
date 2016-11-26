#include <iostream>
#include "CascadeCNN.h"
using namespace std;
using namespace cv;

int main() {

    string mean_file = "../Models/imagenet_mean.binaryproto";

    vector<string> model_file = {
        "../Models/12c/deploy.prototxt",
        "../Models/12cal/deploy.prototxt",
        "../Models/24c/deploy.prototxt",
        "../Models/24cal/deploy.prototxt",
        "../Models/48c/deploy.prototxt",
        "../Models/48cal/deploy.prototxt"
    };

    vector<string> trained_file = {
            "../Models/12c/12c.caffemodel",
            "../Models/12cal/12cal.caffemodel",
            "../Models/24c/24c.caffemodel",
            "../Models/24cal/24cal.caffemodel",
            "../Models/48c/48c.caffemodel",
            "../Models/48cal/48cal.caffemodel"
    };

    vector<Rect> rectangles;

    string img_path = "../result/trump.jpg";
    Mat img = imread(img_path);


    CascadeCNN cascadeCNN(model_file,trained_file,mean_file);

//    cascadeCNN.timer_begin();
//    cascadeCNN.detection_test(img, rectangles);
    cascadeCNN.detection(img, rectangles);
//    cascadeCNN.timer_end();

    for(int i = 0; i < rectangles.size(); i++)
        rectangle(img, rectangles[i], Scalar(255, 0, 0));
    imshow("face", img);
    waitKey(0);

    return 0;
}

//int main(int, char**)
//{
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