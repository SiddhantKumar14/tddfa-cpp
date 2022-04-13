#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "detector.h"
#include "priorbox.h"
#include <fstream>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    string model_path = argv[1];
    string image_path = argv[2];
    Mat clr = imread(image_path,IMREAD_COLOR);
    Mat image;
    Mat conf;
    Mat loc;
    resize(clr, image, Size(), 1.0, 1.0);
    ParamConfig config = {0.8f, 0.3f, model_path};
    vector<vector<float>> m_sizes = {{32, 64, 128}, {256}, {512}};
    vector<float> steps = {32.0, 64.0, 128.0};
    vector<float> variance = {0.1, 0.2};
    PriorBox pbox(Size(image.cols, image.rows), m_sizes, steps, false);
    Mat anchors;
    pbox.forward(anchors);
    Detector detector(config);
    detector.detect(image, loc, conf);
    auto priors = pbox.decode(loc, anchors, variance, image.cols, image.rows);

    cout << priors.row(15349) << endl;
    detector.postProcess(priors, conf, clr);
    imshow("detection", clr);
    imwrite("assert/output.jpg", clr);
    waitKey(0);
}
