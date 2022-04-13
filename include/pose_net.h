#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

struct Config
{
    float confThreshold;
    float nmsThreshold;
    string model_path;
};

class TDDFA
{
public:
    TDDFA(Config& config);
    ~TDDFA();
    void detect(Mat& image, vector<Rect> box, Mat& conf);

private:
    float nmsThreshold = 0.3;
    float confThreshold = 0.8;
    dnn::Net net;
};
