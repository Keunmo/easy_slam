#include <string>
#include <thread>
#include <map>
#include <unordered_map>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <highfive/highfive.hpp>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include "FeatureExtractor.h"

namespace py = pybind11;

namespace EASY_SLAM
{
FeatureExtractor::FeatureExtractor(const std::string &strSettingsFile, const eFeatureType featureType, const bool asHalf)
:mFeatureType(featureType){
    cv::FileStorage fSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if (!fSettings.isOpened()){
        std::cerr << "Failed to open settings file at: " << strSettingsFile << std::endl;
        exit(-1);
    }
    if (mFeatureType == RELOCFEAT){
        mFeatureConf = static_cast<std::string>(fSettings["RelocFeature"]);
    }
    else if (mFeatureType == GLOBALFEAT){
        mFeatureConf = static_cast<std::string>(fSettings["GlobalFeature"]);
    }
    if (mFeatureConf == "ORB"){
        // use opencv orb extractor
        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float scaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        // int iniThFAST = fSettings["ORBextractor.iniThFAST"];
        // int minThFAST = fSettings["ORBextractor.minThFAST"];
        mpORBextractor = cv::ORB::create(nFeatures, scaleFactor, nLevels);
    }
    if else (mFeatureConf == "DBoW3"){ // TODO
        // load vocabulary
        std::string strVocFile = fSettings["VocFile"];
        std::cout << "Loading vocabulary from: " << strVocFile << std::endl;
        mpORBvoc = new DBoW3::Vocabulary(strVocFile);
        mpDbowdb = new DBoW3::Database(*mpORBvoc, false, 0); // false = do not use direct index
    }
    else {
        // bind python script to extract features
        // initialize python FeatureExtractor class 
        py::initialize_interpreter();
        py::module feature_extractor_module = py::module::import("feature_extractor");
        pyFeatureExtractor = feature_extractor_module.attr("FeatureExtractor")(mFeatureConf, featureType, asHalf);
    }
}

FeatureExtractor::~FeatureExtractor(){
    py::finalize_interpreter();
}

FeatureExtractor::Extract(const cv::Mat &im, const long unsigned int frameId, const bool save, std::vector<cv::KeyPoint>& kp, cv::Mat& des){
    if (mFeatureConf == "ORB" | mFeatureConf == "DBoW3"){
        // use opencv orb extractor
        mpORBextractor->detectAndCompute(im, cv::noArray(), kp, des);
        // save descriptors to hdf5 file. 
        // file format: 
        if (save){

        }
    }
    else {
        // extract features using python script
        py::object pyImage = py::cast(im);
        py::object pyFrameId = py::cast(frameId);
        py::object pySave = py::cast(save);
        py::dict pyPred = pyFeatureExtractor.attr("extract")(pyImage, pyFrameId, pySave);
        // for (auto item : pyPred){
        //     (*pred)[item.first.cast<std::string>()] = item.second;
        // }
        py::array_t<py::float_> pyKeypoints = pyPred["keypoints"].cast<py::array_t<py::float_>>();
        auto kpbuf = pyKeypoints.request();
        float *ptr = static_cast<float *>(kpbuf.ptr);
        size_t num_kp = kpbuf.shape[0];
        kp->clear();
        kp->reserve(num_kp);
        for (size_t i = 0; i < num_kp; i++){
            float x = ptr[i * 2];
            float y = ptr[i * 2 + 1];
            kp->emplace_back(cv::KeyPoint(x, y, 1.0f));
        }

        py::array_t<py::float_> pyDescriptors = pyPred["descriptors"].cast<py::array_t<py::float_>>();
        auto desbuf = pyDescriptors.request();
        int rows = desbuf.shape[0]; // desc dim
        int cols = desbuf.shape[1]; // num kps
        std::cout << "rows: " << rows << ", cols: " << cols << std::endl; // for debugging
        des = cv::Mat(cols, rows, CV_16F, desbuf.ptr).clone();
        std::cout << "des size: " << des.size() << std::endl; // for debugging
    }
}
} // namespace EASY_SLAM