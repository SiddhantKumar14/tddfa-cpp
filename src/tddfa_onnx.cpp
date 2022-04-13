#include <iostream>
#include <string>
#include <vector>
#include <opencv2/dnn/dnn.hpp>

//TODO: define Kvs with params, create kvs obj, depickle config files, save as txt/config files, create tddfa obj

using namespace std;

class Kvs {
    public:
    bool gpu_mode = false;
    int gpu_id = 0;
    int size = 120;
    string param_mean_std_fp = "headpose//configs//param_mean_std_62d_120x120.pkl";
    string checkpoint_fp = "weights//mb05_120x120.onnx";
    string onnx_fp = checkpoint_fp;
    Mat param_mean[62];
    Mat param_std[62]; // read from pkl
    string crop_policy = "box";
};

class TDDFA {
    Kvs kvs;
    // start onnxruntime session
    vector call(Mat img, Mat objs){
        vector <vector <int>> param_lst[10], roi_box_lst[10];
        int i = 0;
        for(auto obj = objs.begin(); obj != objs.end(); objs ++) {
            i += 1;
            float param_list[62], roi_box[4];
            Mat resized;
            Mat blob;

            roi_box = parse_roi_box_from_bbox(obj);
            roi_box_lst.push_back(roi_box);

            Mat cropped_image = crop_image(img, roi_box);
            cv::resize(img, resized, Size(kvs.size, kvs.size), INTER_LINEAR);

            //scalar 
            dnn::blobFromImage(img, blob, 1.0/ 128, Size(resized.cols, resized.rows), scalar(127.5, 127, 127), true, false, CV_32F);
            
            this->net.setInput(blob);
            std::vector<string> outLayerNames = this->net.getUnconnectedOutLayersNames();
            std::vector<Mat> outs;
            this->net.forward(outs, outLayerNames);

            for (int i = 0; i < outs.size(); i++){
                if (i == 0)
                    Mat(outs[i].size[1], outs[i].size[2], CV_32F, outs[i].data).copyTo(loc);
                else
                    Mat(outs[i].size[1], outs[i].size[2], CV_32F, outs[i].ptr<float>()).copyTo(conf);
            }

            param = outs //after postprocessing
            param = session.run // onnx 
            param = Flatten(param);
            for (int i = 0; i < 62; i++)
            {
                param[i] = param_mean[i] + param_std[i] * param[i];
            }
        } 
    }
};

int main() {
    if (__cplusplus == 201703L) std::cout << "C++17\n";
    else if (__cplusplus == 201402L) std::cout << "C++14\n";
    else if (__cplusplus == 201103L) std::cout << "C++11\n";
    else if (__cplusplus == 199711L) std::cout << "C++98\n";
    else std::cout << "pre-standard C++\n";
};
