
#include "pose_net.h"

TDDFA::TDDFA(Config &config)
{
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->net = dnn::readNetFromONNX(config.model_path);
    this->net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(dnn::DNN_TARGET_CPU); //not exact params
}

TDDFA::~TDDFA()
{
    this->net.~Net();
}

void TDDFA::detect(Mat &img, vector<Rect> box, Mat &conf) // just copied from detector.h
{
    Mat blob;
    dnn::blobFromImage(img, blob, 1.0, Size(120,120), Scalar(127.5, 127, 127), true, false, CV_32F);
    this->net.setInput(blob);
    std::vector<string> outLayerNames = this->net.getUnconnectedOutLayersNames();
    std::vector<Mat> outs;
    this->net.forward(outs, outLayerNames);

}