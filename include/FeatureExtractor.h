#include <string>
#include <thread>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace EASY_SLAM
{

class FeatureExtractor
{
public:
    enum eFeatureType{
        RELOCFEAT=0,
        GLOBALFEAT=1,
    };

public:
    FeatureExtractor(const std::string &strSettingsFile, const eFeatureType featureType, const bool asHalf = true);

    // extractor for local features
    // local feature: ['descriptors', 'image_size', 'keypoint_scores', 'keypoints']
    // void Extract(const cv::Mat &im, const std::string frameId, const bool save = false);
    // void Extract(const cv::Mat &im, const std::string frameId, const bool save = false, std::unordered_map<std::string, py::object>* pred);
    void Extract(const cv::Mat &im, const long unsigned int frameId, const bool save = false, std::vector<cv::KeyPoint>& kp, cv::Mat& des);

    // extractor for global features
    // global feature: ['global_descriptor', 'image_size']

private:
    eFeatureType mFeatureType;
    std::string mFeatureConf;
    std::string mFeaturePath;
    cv::ORB *mpORBextractor;
    py::object pyFeatureExtractor;


}; // class FeatureExtractor

} // namespace EASY_SLAM
