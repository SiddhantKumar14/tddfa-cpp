#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

struct ParamConfig
{
    float confThreshold;
    float nmsThreshold;
    string model_path;
};

class Detector
{
public:
    Detector(ParamConfig &config);
    ~Detector();
    void detect(Mat &image, Mat &loc, Mat &conf);
    void postProcess(Mat &boxes, Mat &conf, Mat &img);

private:
    float nmsThreshold=0.3;
    float confThreshold=0.8;
    dnn::Net net;
};
