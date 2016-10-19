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
        std::shared_ptr<Net<float>> net;
        cv::Size input_geometry;
        int num_channel;

        net.reset(new Net<float>(model_file[i], TEST));
        net->CopyTrainedLayersFrom(trained_file[i]);

        Blob<float>* input_layer = net->input_blobs()[0];
        num_channel = input_layer->channels();
        input_geometry = cv::Size(input_layer->width(), input_layer->height());

//        net12c_ = net;
        nets_.push_back(net);
        input_geometry_.push_back(input_geometry);
        if(i == 0)
            num_channels_ = num_channel;
        else if(num_channels_ != num_channel)
            std::cout << "Error: The number channels of the net are different!" << std::endl;
    }

    SetMean(mean_file);
}

void CascadeCNN::detection(const cv::Mat &img, std::vector<cv::Rect> &rectangles)
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

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample.convertTo(sample_float, CV_32FC3);
    else
        sample.convertTo(sample_float, CV_32FC1);

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

void CascadeCNN::detect_net(int i)
{
    std::shared_ptr<Net<float>> net = nets_[i];

    std::vector<cv::Rect> rectangles;
    std::vector<float> confidence;
    //cv::Size scale = cv::Size(48, 48);
    for(int j = 0; j < rectangles_.size(); j++)
    {
        cv::Mat img = crop(img_, rectangles_[j]);
        if (img.size() != input_geometry_[i])
            cv::resize(img, img, input_geometry_[i]);

        std::vector<cv::Mat> input_channels;
        std::vector<float> prediction = Predict(img, i);
        float conf = prediction[1];

        if(conf > threshold_confidence_)
        {
            rectangles.push_back(rectangles_[j]);
            confidence.push_back(conf);
        }
    }

    rectangles_ = rectangles;
    confidence_ = confidence;
}

void CascadeCNN::calibrate_net(int i)
{
    std::shared_ptr<Net<float>> net = nets_[i];

    std::vector<cv::Rect> rectangles;
    std::vector<float> confidence;

    for(int j = 0; j < rectangles_.size(); j++)
    {
        cv::Mat img = crop(img_, rectangles_[j]);
        if (img.size() != input_geometry_[i])
            cv::resize(img, img, input_geometry_[i]);

        std::vector<cv::Mat> input_channels;
        std::vector<float> prediction = Predict(img, i);

        calibrate(prediction, j);
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

    net->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

void CascadeCNN::WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i)
{
    Blob<float>* input_layer = nets_[i]->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(img, *input_channels);

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
