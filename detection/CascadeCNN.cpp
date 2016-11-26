//
// Created by Yang on 16/10/17.
//

//#include <bits/shared_ptr.h>
#include "CascadeCNN.h"


CascadeCNN::CascadeCNN() {}

CascadeCNN::CascadeCNN(const std::vector<std::string> model_file,
                       const std::vector<std::string> trained_file,
                       const std::string &mean_file)
{
    #ifdef CPU_ONLY
        Caffe::set_mode(Caffe::CPU);
    #else
        Caffe::set_mode(Caffe::GPU);
    #endif

    for(int i = 0; i < model_file.size(); i++)
    {
        std::shared_ptr<Net<float>> net;
        cv::Size input_geometry;
        int num_channel;

        net.reset(new Net<float>(model_file[i], TEST));
        net->CopyTrainedLayersFrom(trained_file[i]);

        Blob<float>* input_layer = net->input_blobs()[0];
        num_channel = input_layer->channels();
        input_geometry = cv::Size(input_layer->width(), input_layer->height());

        nets_.push_back(net);
        input_geometry_.push_back(input_geometry);
        if(i == 0)
            num_channels_ = num_channel;
        else if(num_channels_ != num_channel)
            std::cout << "Error: The number channels of the net are different!" << std::endl;
    }
    mean_file_ = mean_file;
    SetMean(mean_file);
}

CascadeCNN::~CascadeCNN() {}

void CascadeCNN::detection_test(const cv::Mat img, std::vector<cv::Rect> &rectangles)
{
    Preprocess(img);
    detect_12c_net();
    //detect_12c_net_test();
    int a = rectangles_.size();
    img_show(img, "0");
    cal_12c_net();
    int b = rectangles_.size();
    img_show(img, "1");
    local_NMS();
    int c = rectangles_.size();
    img_show(img, "2");
    detect_24c_net();
    int d = rectangles_.size();
    img_show(img, "3");
    cal_24c_net();
    int e = rectangles_.size();
    img_show(img, "4");
    local_NMS_test();
    int f = rectangles_.size();
    img_show(img, "5");
    detect_48c_net();
    int g = rectangles_.size();
    img_show(img, "6");
    global_NMS();
    int h = rectangles_.size();
    img_show(img, "7");
    cal_48c_net();
    int i = rectangles_.size();
    img_show(img, "8");
    //global_NMS_specify();
    //int J = rectangles_.size();
    //img_show(img, "9");

    rectangles = rectangles_;
}

void CascadeCNN::detection(const cv::Mat &img, std::vector<cv::Rect> &rectangles)
{
    Preprocess(img);
    detect_12c_net();
    cal_12c_net();
    local_NMS();
    detect_24c_net();
    cal_24c_net();
    //local_NMS();
    local_NMS_test();
    detect_48c_net();
    global_NMS();
    cal_48c_net();
    global_NMS_specify();

    rectangles = rectangles_;
}

void CascadeCNN::detection(const cv::Mat img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence)
{
    Preprocess(img);
    detect_12c_net();
    cal_12c_net();
    local_NMS();
    detect_24c_net();
    cal_24c_net();
    //local_NMS();
    local_NMS_test();
    detect_48c_net();
    global_NMS();
    cal_48c_net();
    global_NMS_specify();

    rectangles = rectangles_;
    confidence = confidence_;
}

void CascadeCNN::SetMean(const std::string& mean_file)
{
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    for(int i = 0; i < nets_.size(); i++)
    {
        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);

        if(mean_blob.channels() != num_channels_)
            std::cout << "Number of channels of mean file doesn't match input layer." << std::endl;

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        cv::Scalar channel_mean = cv::mean(mean);
        mean = cv::Mat(input_geometry_[i], mean.type(), channel_mean);

        mean_.push_back(mean);
    }

}

cv::Mat CascadeCNN::SetMean(cv::Mat img, const std::string& mean_file)
{
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);


    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);

    if(mean_blob.channels() != num_channels_)
        std::cout << "Number of channels of mean file doesn't match input layer." << std::endl;

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    cv::Size img_geometry;
    img_geometry = cv::Size(img.cols, img.rows);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean = cv::Mat(img_geometry, mean.type(), channel_mean);

    return mean;

}

void CascadeCNN::Preprocess(const cv::Mat &img)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample.convertTo(sample_float, CV_32FC3);
    else
        sample.convertTo(sample_float, CV_32FC1);

    /*
     * mean file is the same dimension with the figure
     */
    cv::Mat sample_normalized;
    cv::Mat img_mean = SetMean(img, mean_file_);
    cv::subtract(sample_float, img_mean, sample_normalized);

    img_ = sample_float;
}

void CascadeCNN::detect_12c_net()
{
    generate_init_rectangles();
//    detect_net_12(0);
    detect_net_batch(0);
}

void CascadeCNN::cal_12c_net()
{
    //calibrate_net(1);
    calibrate_net_batch(1);
    clear();
}

void CascadeCNN::detect_24c_net()
{
    //detect_net(2);
    detect_net_batch(2);
}

void CascadeCNN::cal_24c_net()
{
    //calibrate_net(3);
    calibrate_net_batch(3);
}

void CascadeCNN::detect_48c_net()
{
    //detect_net(4);
    detect_net_batch(4);
}

void CascadeCNN::cal_48c_net()
{
    //calibrate_net(5);
    calibrate_net_batch(5);
}

void CascadeCNN::local_NMS()
{
    std::vector<cv::Rect> cur_rects = rectangles_;
    std::vector<float> confidence = confidence_;
    float threshold = threshold_NMS_;

    for(int i = 0; i < cur_rects.size(); i++)
    {
        for(int j = i + 1; j < cur_rects.size(); )
        {
            if(IoU(cur_rects[i], cur_rects[j]) > threshold)
            {
                float a = IoU(cur_rects[i], cur_rects[j]);
//                if(confidence[i] == confidence[j])
//                {
//                    cur_rects.erase(cur_rects.begin() + j);
//                    confidence.erase(confidence.begin() + j);
//                }
                if(confidence[i] >= confidence[j])
                {
                    cur_rects.erase(cur_rects.begin() + j);
                    confidence.erase(confidence.begin() + j);
                }
                else
                {
                    cur_rects.erase(cur_rects.begin() + i);
                    confidence.erase(confidence.begin() + i);
                    i--;
                    break;
                }
            }
            else
            {
                j++;
            }

        }
    }

    rectangles_ = cur_rects;
    confidence_ = confidence;

}

void CascadeCNN::local_NMS_test()
{
    std::vector<cv::Rect> cur_rects = rectangles_;
    std::vector<float> confidence = confidence_;
    float threshold = threshold_NMS_;
    float threshold_IoM = threshold_NMS_ * 2;

    for(int i = 0; i < cur_rects.size(); i++)
    {
        for(int j = i + 1; j < cur_rects.size(); )
        {
            if(IoU(cur_rects[i], cur_rects[j]) > threshold || IoM(cur_rects[i], cur_rects[j]) > threshold_IoM)
            {
                float a = IoU(cur_rects[i], cur_rects[j]);
//                if(confidence[i] == confidence[j])
//                {
//                    cur_rects.erase(cur_rects.begin() + j);
//                    confidence.erase(confidence.begin() + j);
//                }
                if(confidence[i] >= confidence[j])
                {
                    cur_rects.erase(cur_rects.begin() + j);
                    confidence.erase(confidence.begin() + j);
                }
                else
                {
                    cur_rects.erase(cur_rects.begin() + i);
                    confidence.erase(confidence.begin() + i);
                    i--;
                    break;
                }
            }
            else
            {
                j++;
            }

        }
    }

    rectangles_ = cur_rects;
    confidence_ = confidence;

}

void CascadeCNN::global_NMS()
{
    std::vector<cv::Rect> cur_rects = rectangles_;
    std::vector<float> confidence = confidence_;
    float threshold_IoM = threshold_NMS_;
    float threshold_IoU = threshold_NMS_ - 0.1;


    for(int i = 0; i < cur_rects.size(); i++)
    {
        for(int j = i + 1; j < cur_rects.size(); )
        {
            if(IoU(cur_rects[i], cur_rects[j]) > threshold_IoU || IoM(cur_rects[i], cur_rects[j]) > threshold_IoM)
            {
                if(confidence[i] >= confidence[j] && confidence[j] < 0.85) //if confidence[i] == confidence[j], it keeps the small one
                {
                    cur_rects.erase(cur_rects.begin() + j);
                    confidence.erase(confidence.begin() + j);
                }
                else if(confidence[i] < confidence[j] && confidence[i] < 0.85)
                {
                    cur_rects.erase(cur_rects.begin() + i);
                    confidence.erase(confidence.begin() + i);
                    i--;
                    break;
                }
                else
                {
                    j++;
                }
            }
            else
            {
                j++;
            }
        }
    }

    rectangles_ = cur_rects;
    confidence_ = confidence;
}

/*
 * for now global_NMS_specify() is completely same with global_NMS()
 * the condition need to change after test
 */
void CascadeCNN::global_NMS_specify()
{
    std::vector<cv::Rect> cur_rects = rectangles_;
    std::vector<float> confidence = confidence_;
    float threshold_IoM = threshold_NMS_;
    float threshold_IoU = threshold_NMS_ - 0.1;

    for(int i = 0; i < cur_rects.size(); i++)
    {
        for(int j = i + 1; j < cur_rects.size(); )
        {
            //the condition need to change after test
            if(IoU(cur_rects[i], cur_rects[j]) > threshold_IoU || IoM(cur_rects[i], cur_rects[j]) > threshold_IoM)
            {
                if(confidence[i] >= confidence[j] && confidence[j] < 0.85) //if confidence[i] == confidence[j], it keeps the small one
                {
                    cur_rects.erase(cur_rects.begin() + j);
                    confidence.erase(confidence.begin() + j);
                }
                else if(confidence[i] >= confidence[j] && confidence[i] < 0.85)
                {
                    cur_rects.erase(cur_rects.begin() + i);
                    confidence.erase(confidence.begin() + i);
                    break;
                }
                else
                {
                    j++;
                }
            }
            else
            {
                j++;
            }
        }
    }

    rectangles_ = cur_rects;
    confidence_ = confidence;
}


void CascadeCNN::generate_init_rectangles()
{
    int dimension = dimension_;
    std::vector<cv::Rect> rects;
    float scale = scale_factor_;
    int strip = dimension_ * strip_ / 12;
    //int small_face_size = small_face_size_;
    //float current_scale = float(small_face_size) / dimension;

    while(dimension < img_.rows && dimension < img_.cols)
    {
        for(int i = 0; i < img_.cols - dimension; i += strip)
        {
            for(int j = 0; j < img_.rows - dimension; j += strip)
            {
                cv::Rect cur_rect(i, j, dimension, dimension);
                rects.push_back(cur_rect);
            }
        }
        dimension = dimension * scale;
        strip = strip * scale;
    }

    rectangles_ = rects;

    std::vector<float> confidence(rects.size());
    confidence_ = confidence;
}


void CascadeCNN::detect_net(int i)
{
    std::shared_ptr<Net<float>> net = nets_[i];
    float threshold_confidence = threshold_confidence_ / 2 ;
    std::vector<cv::Rect> rectangles;
    std::vector<float> confidence;
    //cv::Size scale = cv::Size(48, 48);

    if(rectangles_.size() == 0)
        return;

    for(int j = 0; j < rectangles_.size(); j++)
    {
        cv::Mat img = crop(img_, rectangles_[j]);
        if (img.size() != input_geometry_[i])
            cv::resize(img, img, input_geometry_[i]);

        //std::vector<cv::Mat> input_channels;
        std::vector<float> prediction = Predict(img, i);
        float conf = prediction[1];

        if(conf > threshold_confidence)
        {
            rectangles.push_back(rectangles_[j]);
            confidence.push_back(conf);
        }
    }

    rectangles_ = rectangles;
    confidence_ = confidence;
}

void CascadeCNN::detect_net_batch(int i) {
    std::shared_ptr<Net<float>> net = nets_[i];
    Blob<float> *input_layer = nets_[i]->input_blobs()[i];
//    int num = input_layer->num();
    float threshold_confidence = threshold_confidence_ / 2;
    std::vector<cv::Rect> rectangles;
    std::vector<float> confidence;
    vector<cv::Mat> cur_imgs;

    if(rectangles_.size() == 0)
        return;

    for (int j = 0; j < rectangles_.size(); j++) {
        cv::Mat img = crop(img_, rectangles_[j]);
        if (img.size() != input_geometry_[i])
            cv::resize(img, img, input_geometry_[i]);
        cur_imgs.push_back(img);
    }

    std::vector<float> prediction = Predict(cur_imgs, i);

    float conf;

    for(int j = 0; j < prediction.size() / 2; j++)
    {
        conf = prediction[2*j+1];
        if (conf > threshold_confidence) {
            rectangles.push_back(rectangles_[j]);
            confidence.push_back(conf);
        }
    }

    cur_imgs.clear();

//        if(j % num == 0 || j == rectangles_.size())
//        {
//            std::vector<float> prediction = Predict(cur_imgs, i);
//            float conf;
//
//            for(int k = 0; k < prediction.size() / 2; k++)
//            {
//                conf = prediction[2*k+1];
//                if (conf > threshold_confidence) {
//                    rectangles.push_back(rectangles_[j]);
//                    confidence.push_back(conf);
//                }
//            }
//
//            cur_imgs.clear();
//        }

    rectangles_ = rectangles;
    confidence_ = confidence;
}

void CascadeCNN::calibrate_net(int i)
{
    std::shared_ptr<Net<float>> net = nets_[i];

    std::vector<cv::Rect> rectangles;
    std::vector<float> confidence;

    if(rectangles_.size() == 0)
        return;

    for(int j = 0; j < rectangles_.size(); j++)
    {
        cv::Mat img = crop(img_, rectangles_[j]);
        if (img.size() != input_geometry_[i])
            cv::resize(img, img, input_geometry_[i]);

        //std::vector<cv::Mat> input_channels;
        std::vector<float> prediction = Predict(img, i);

        calibrate(prediction, j);
    }
}

void CascadeCNN::calibrate_net_batch(int i)
{
    std::shared_ptr<Net<float>> net = nets_[i];

    std::vector<cv::Rect> rectangles;
    std::vector<float> confidence;
    std::vector<cv::Mat> imgs;
    int index_cal = 0;

    int a = rectangles_.size();
    if(rectangles_.size() == 0)
        return;

    for(int j = 0; j < rectangles_.size(); j++)
    {
        cv::Mat img = crop(img_, rectangles_[j]);
        if (img.size() != input_geometry_[i]) {
            cv::resize(img, img, input_geometry_[i]);
        }
        imgs.push_back(img);
        //std::vector<cv::Mat> input_channels;
    }
    std::vector<float> prediction = Predict(imgs, i);
    std::vector<float> cur_pred;
    for(int k = 0; k < imgs.size(); k++)
    {
        cur_pred = vector<float>(prediction.begin(), prediction.begin()+45 );
        prediction.erase(prediction.begin(), prediction.begin()+45);
        calibrate(cur_pred, k);
        cur_pred.clear();
    }

}



void CascadeCNN::calibrate(std::vector<float> prediction, int j)
{
    std::vector<int> index;

    for(int i = 0; i < prediction.size(); i++)
    {
        if(prediction[i] > threshold_confidence_)
            index.push_back(i);
    }

    if(index.size() == 0)
        return;

    float x_change = 0, y_change = 0, s_change = 0;

    for(int i = 0; i < index.size(); i++)
    {
        int cur_index = index[i];

        if (cur_index >= 0 && cur_index <= 8)
            s_change += 0.83;
        else if (cur_index >= 9 && cur_index <= 17)
            s_change += 0.91;
        else if (cur_index >= 18 && cur_index <= 26)
            s_change += 1.0;
        else if (cur_index >= 27 && cur_index <= 35)
            s_change += 1.10;
        else
            s_change += 1.21;

        if (cur_index % 9 <= 2)
            x_change += -0.17;
        else if (cur_index % 9 >= 6 && cur_index % 9 <= 8)
            x_change += 0.17;

        if (cur_index % 3 == 0)
            y_change += -0.17;
        else if (cur_index % 3 == 2)
            y_change += 0.17;
    }

    //calculate the mean of total change
    s_change = s_change / index.size();
    x_change = x_change / index.size();
    y_change = y_change / index.size();

    cv::Rect rect;
    rect.x = std::max(0, int(rectangles_[j].x - rectangles_[j].width * x_change / s_change));
    rect.y = std::max(0, int(rectangles_[j].y - rectangles_[j].height * y_change / s_change));
    rect.width = std::min(img_.cols - rectangles_[j].x, int(rectangles_[j].width / s_change));
    rect.height = std::min(img_.rows - rectangles_[j].y, int(1.1 * rectangles_[j].height / s_change));

    rectangles_[j] = rect;

    //something need to test
    /*
     * this function related to the confidence value in the next NMS step,
     * it will decide which element should be delete
     */
//    std::vector<float> pred;
//    for(int i = 0; i < index.size(); i++)
//    {
//        pred.push_back(prediction[index[i]]);
//    }
//    float conf = *std::max_element(pred.begin(), pred.end());
//    confidence_[j] = conf;

}

void CascadeCNN::clear()
{
    for(int i = 0; i < rectangles_.size(); )
    {
        if(confidence_[i] == 0)
        {
            rectangles_.erase(rectangles_.begin() + i);
            confidence_.erase(confidence_.begin() + i);
        }
        else
            i++;
    }
}
float CascadeCNN::IoU(cv::Rect rect1, cv::Rect rect2)
{
    int x_overlap, y_overlap, intersection, unions;
    x_overlap = std::max(0, std::min((rect1.x + rect1.width), (rect2.x + rect2.width)) - std::max(rect1.x, rect2.x));
    y_overlap = std::max(0, std::min((rect1.y + rect1.height), (rect2.y + rect2.height)) - std::max(rect1.y, rect2.y));
    intersection = x_overlap * y_overlap;
    unions = rect1.width * rect1.height + rect2.width * rect2.height - intersection;
    return float(intersection)/unions;
}

float CascadeCNN::IoM(cv::Rect rect1, cv::Rect rect2)
{
    int x_overlap, y_overlap, intersection, min_area;
    x_overlap = std::max(0, std::min((rect1.x + rect1.width), (rect2.x + rect2.width)) - std::max(rect1.x, rect2.x));
    y_overlap = std::max(0, std::min((rect1.y + rect1.height), (rect2.y + rect2.height)) - std::max(rect1.y, rect2.y));
    intersection = x_overlap * y_overlap;
    min_area = std::min((rect1.width * rect1.height), (rect2.width * rect2.height));
    return float(intersection)/min_area;
}

std::vector<float> CascadeCNN::Predict(const cv::Mat& img, int i)
{
    std::shared_ptr<Net<float>> net = nets_[i];

    Blob<float>* input_layer = net->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_[i].height, input_geometry_[i].width);
    /* Forward dimension change to all layers. */
    net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(img, &input_channels, i);
    //WrapInputLayer(img, i);
    net->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

std::vector<float> CascadeCNN::Predict(const std::vector<cv::Mat> imgs, int i)
{
    std::shared_ptr<Net<float>> net = nets_[i];

    Blob<float>* input_layer = net->input_blobs()[0];
    input_layer->Reshape(imgs.size(), num_channels_,
                         input_geometry_[i].height, input_geometry_[i].width);
    int num = input_layer->num();
    /* Forward dimension change to all layers. */
    net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapBatchInputLayer(imgs, &input_channels, i);
    //WrapInputLayer(img, &input_channels, i);

    net->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels() * output_layer->num();
    return std::vector<float>(begin, end);
}

void CascadeCNN::WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i)
{
    Blob<float>* input_layer = nets_[i]->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int j = 0; j < input_layer->channels(); ++j)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }


    //cv::Mat sample_normalized;
    //cv::subtract(img, mean_[i], img);
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(img, *input_channels);

}

/*
 * When I create the std::vector<cv::Mat> *input_channels in the function, it can not work
 * I need to find the reason of that
 */
void CascadeCNN::WrapInputLayer(const cv::Mat& img, int i)
{
    Blob<float>* input_layer = nets_[i]->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();

    std::vector<cv::Mat> *input_channels;
    for (int j = 0; j < input_layer->channels(); ++j)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }


    //cv::Mat sample_normalized;
    //cv::subtract(img, mean_[i], img);
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(img, *input_channels);

}

void CascadeCNN::WrapBatchInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i) {
    Blob<float> *input_layer = nets_[i]->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float *input_data = input_layer->mutable_cpu_data();

    for (int j = 0; j < num; j++) {
        //std::vector<cv::Mat> *input_channels;
        for (int k = 0; k < input_layer->channels(); ++k) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels->push_back(channel);
            input_data += width * height;
        }
        cv::Mat img = imgs[j];
        cv::split(img, *input_channels);
        input_channels->clear();
    }
}

cv::Mat CascadeCNN::crop(cv::Mat img, cv::Rect rect)
{
    if(rect.x <= 0) rect.x = 0;
    if(rect.y <= 0) rect.y = 0;
    if(img.cols < (rect.x + rect.width)) rect.width = img.cols-rect.x;
    if(img.rows < (rect.y + rect.height)) rect.height = img.rows - rect.y;
    if(rect.width<0)
    {
        rect.x=0;
        rect.width = 0;
    }
    if(rect.height<0)
    {
        rect.y=0;
        rect.height = 0;
    }
    return img(rect);
}

void CascadeCNN::img_show(cv::Mat img, std::string name)
{
    cv::Mat img_show;
    img.copyTo(img_show);
    //cv::imwrite("/home/xileli/Documents/program/CascadeCNN/" + name + "test.jpg", img);
    for(int i = 0; i < rectangles_.size(); i++)
    {
        rectangle(img_show, rectangles_[i], cv::Scalar(255, 0, 0));
        cv::putText(img_show, std::to_string(confidence_[i]), cvPoint(rectangles_[i].x + 3, rectangles_[i].y + 13),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255, 0, 0), 1, CV_AA);
    }

    cv::imwrite("../result/" + name + ".jpg", img_show);
    //cv::waitKey(0);
}

void CascadeCNN::timer_begin()
{
    time_begin_ = std::chrono::high_resolution_clock::now();
}

void CascadeCNN::timer_end()
{
    time_end_ = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::milliseconds> (time_end_ - time_begin_);
    record(time_span.count());
}

void CascadeCNN::record(double num)
{
    std::fstream file("../result/record.txt", ios::app);
    std::cout << num << std::endl;
    file << num << std::endl;
    file.close();
}
