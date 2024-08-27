#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void extract(const cv::Mat& img, cv::Mat& des, vector<cv::KeyPoint>& kp) {
    const static auto& orb = cv::ORB::create();
    orb->detectAndCompute(img, cv::noArray(), kp, des);
}

int main(int argc, const char * argv[]) {
    // load image
    cv::Mat img = cv::imread("/home/keunmo/workspace/Hierarchical-Localization/datasets/sacre_coeur/mapping/02928139_3448003521.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Failed to load image" << endl;
        return -1;
    }
    // extract features
    cv::Mat des;
    vector<cv::KeyPoint> kp;
    extract(img, des, kp);

    // check des shape
    cout << "des shape: " << des.size() << endl;
    cout << "1st des shape: " << des.row(0).size() << endl;
    cout << "kp size: " << kp.size() << endl;
    cout << "1st kp: " << kp[0].pt << endl;

    // make empty 32 x 500 matrix
    cv::Mat des2 = cv::Mat::zeros(500, 32, CV_8UC1);
    cout << "des2 shape: " << des2.size() << endl;

    return 0;
}