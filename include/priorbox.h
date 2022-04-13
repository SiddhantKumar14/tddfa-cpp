#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
using namespace std;
using namespace cv;

class PriorBox
{
public:
    PriorBox(Size image_size, vector<vector<float>> &min_sizes, vector<float> &step, bool clip);
    void forward(Mat&anchors);
    Mat decode(Mat &loc,Mat&priors,vector<float> &variance,int width,int height);
private:
    vector<vector<float>> min_sizes;
    vector<float>  steps;
    bool clip;
    Size image_size;
    vector<vector<float>> feature_maps;
};
