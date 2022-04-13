#include "detector.h"

Detector::Detector(ParamConfig &config)
{
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->net = dnn::readNetFromONNX(config.model_path);
    this->net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(dnn::DNN_TARGET_CPU);
}

Detector::~Detector()
{
    this->net.~Net();
}

void Detector::detect(Mat &img, Mat &loc, Mat &conf)
{
    Mat blob;
    dnn::blobFromImage(img, blob, 1.0, Size(img.cols, img.rows), Scalar(104.0, 117.0, 123.0), false, false);
    this->net.setInput(blob);
    std::vector<string> outLayerNames = this->net.getUnconnectedOutLayersNames();
    std::vector<Mat> outs;
    this->net.forward(outs, outLayerNames);

    for (int i = 0; i < outs.size(); i++)
    {
        if (i == 0)
            Mat(outs[i].size[1], outs[i].size[2], CV_32F, outs[i].data).copyTo(loc);
        else
            Mat(outs[i].size[1], outs[i].size[2], CV_32F, outs[i].ptr<float>()).copyTo(conf);
    }
}
void Detector::postProcess(Mat &priors, Mat &conf, Mat &img)
{
    vector<Rect> boxes;
    vector<float> scores;
    vector<int> indices;
    int ind = 0;
    for (int i = 0; i < priors.rows; i++)
    {
        float b = conf.at<float>(i, 1);
        if (b > 0.05)
        {
            scores.push_back(b);
        }
        else
        {
            continue;
        }
        indices.push_back(ind);
        float xmin = priors.at<float>(i, 0);
        float ymin = priors.at<float>(i, 1);
        float xmax = priors.at<float>(i, 2);
        float ymax = priors.at<float>(i, 3);
        boxes.push_back(Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)));
        ind++;
    }
    dnn::NMSBoxes(boxes, scores, this->confThreshold, this->nmsThreshold, indices);
    // file me store, pickle for cpp jo bhi 
    for (int i = 0; i < indices.size(); i++)
    {
        rectangle(img, boxes[indices[i]], Scalar(0, 255, 0), 2);
        auto box = boxes[indices[i]];
        int y = max(box.y - 10,0);
        char sc[80];
        sprintf(sc,"%.4f", scores[indices[i]]);
        ;
        putText(img, sc, Point(box.x, y), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 2);
    }
}
