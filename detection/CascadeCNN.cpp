//
// Created by 刘阳 on 16/10/17.
//

#include "CascadeCNN.h"


CascadeCNN::CascadeCNN() {}

CascadeCNN::CascadeCNN(const std::vector<std::string> model_file,
                       const std::vector<std::string> trained_file,
                       const std::string &mean_file)
{
    if(mode_pu_ == 0)
    {
        Caffe::set_mode(Caffe::CPU);
    }
    else
    {
        Caffe::set_mode(Caffe::GPU);
    }

    for(int i = 0; i < model_file.size(); i++)
    {
        shared_ptr<Net<float> > net;
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

    SetMean(mean_file);
}

void CascadeCNN::detection(const cv::Mat &img, std::vector<cv::Rect> *rectangles)
{
    Preprocess(img);

    detect_12c_net();
    cal_12c_net();
    local_NMS();
    detect_24c_net();
    cal_24c_net();
    local_NMS();
    detect_48c_net();
    global_NMS();
    cal_48c_net();
    global_NMS_specify();

    rectangles = rectangles_;
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

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    img_ = sample_normalized;
}

void CascadeCNN::detect_12c_net()
{
    genereate_init_rectangles();
    detect_net(0);
}

void CascadeCNN::cal_12c_net()
{
    calibrate_net(1);
}

void CascadeCNN::detect_24c_net()
{
    detect_net(2);
}

void CascadeCNN::cal_24c_net()
{
    calibrate_net(3);
}

void CascadeCNN::detect_48c_net()
{
    detect_net(4);
}

void CascadeCNN::cal_48c_net()
{
    calibrate_net(5);
}

void CascadeCNN::local_NMS()
{
    std::vector<cv::Rect> cur_rects = rectangles_;
    int num_rects = cur_rects.size();
    float threshold = threshold_NMS_;

}

void CascadeCNN::global_NMS()
{
    std::vector<cv::Rect> cur_rects = rectangles_;
    int num_rects = cur_rects.size();
    float threshold = threshold_NMS_;
}

void CascadeCNN::global_NMS_specify()
{
    std::vector<cv::Rect> cur_rects = rectangles_;
    int num_rects = cur_rects.size();
    float threshold = threshold_NMS_;
}

void CascadeCNN::genereate_init_rectangles()
{
    int dimension = 12;
    std::vector<cv::Rect> rects;
    float scale = scale_factor_;
    //int small_face_size = small_face_size_;
    //float current_scale = float(small_face_size) / dimension;

    while(dimension < img_.rows && dimension < img_.cols)
    {
        for(int i = 0; i < img_.cols - dimension; i += dimension)
        {
            for(int j = 0; j < img_.rows - dimension; j += dimension)
            {
                cv::Rect cur_rect(i, j, dimension, dimension);
                rects.push_back(cur_rect);
            }
        }
        dimension = dimension * scale;
    }
}

