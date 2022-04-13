#include "priorbox.h"
#include <opencv2/core.hpp>

PriorBox::PriorBox(Size image_size, vector<vector<float>> &min_sizes, vector<float> &steps, bool clip)
{
    this->min_sizes = min_sizes;
    this->steps = steps;
    this->clip = clip;
    this->image_size = image_size;
    for (int i = 0; i < this->steps.size(); i++)
    {
        this->feature_maps.push_back(
            {
                ceil(image_size.height / this->steps[i]),
                ceil(image_size.width / this->steps[i]),
            });
    }
}
void PriorBox::forward(Mat &anchors)
{
    vector<float> points;
    int length = 0;
    for (int k = 0; k < this->feature_maps.size(); k++)
    {
        vector<float> sizes = this->min_sizes[k];
        for (int i = 0; i < feature_maps[k][0]; i++)
        {
            for (int j = 0; j < feature_maps[k][1]; j++)
            {
                for (auto _size : sizes)
                {
                    float s_kx = _size / this->image_size.width;
                    float s_ky = _size / this->image_size.height;
                    vector<float> dense_cx;
                    vector<float> dense_cy;
                    vector<float> _steps;
                    if (_size == 32)
                    {
                        _steps = {0.0f, 0.25f, 0.5f, 0.75f};
                    }
                    else if (_size == 64)
                    {
                        _steps = {0.0f, 0.5f};
                    }
                    else
                    {
                        _steps = {0.5f};
                    }
                    for (auto step : _steps)
                    {
                        dense_cx.push_back((j + step) * this->steps[k] / this->image_size.width);
                        dense_cy.push_back((i + step) * this->steps[k] / this->image_size.height);
                    }
                    for (int a = 0; a < dense_cy.size(); a++)
                    {
                        for (int b = 0; b < dense_cx.size(); b++)
                        {
                            points.emplace_back(dense_cx[b]);
                            points.emplace_back(dense_cy[a]);
                            points.emplace_back(s_kx);
                            points.emplace_back(s_ky);
                            length++;
                        }
                    }
                }
            }
        }
    }
    Mat mt = Mat(length, 4, CV_32F, points.data());
    mt.copyTo(anchors);
}

Mat PriorBox::decode(Mat &loc, Mat &priors, vector<float> &variance, int width, int height)
{
    Mat pf2 = priors.rowRange(0, priors.rows).colRange(0, 2);
    Mat ps2 = priors.rowRange(0, priors.rows).colRange(2, 4);
    Mat lf2 = loc.rowRange(0, loc.rows).colRange(0, 2);
    Mat ls2 = loc.rowRange(0, loc.rows).colRange(2, 4);

    pf2 = pf2 + (lf2 * variance[0]).mul(ps2);
    Mat lexp = ls2 * variance[1];
    cv::exp(lexp, lexp);
    ps2 = ps2.mul(lexp);
    pf2 = pf2 - ps2 / 2;
    ps2 = ps2 + pf2;
    cout << ps2.row(0) << endl;
    cout << pf2.row(0) << endl;

    Mat p1 = pf2.rowRange(0, pf2.rows).colRange(0, 1) * width / 2.5;
    Mat p2 = pf2.rowRange(0, pf2.rows).colRange(1, 2) * height / 2.5;
    Mat p3 = ps2.rowRange(0, ps2.rows).colRange(0, 1) * width / 2.5;
    Mat p4 = ps2.rowRange(0, ps2.rows).colRange(1, 2) * height / 2.5;
    vector<Mat> mats = {p1, p2, p3, p4};
    Mat rst;
    hconcat(mats, rst);
    return rst;
}
