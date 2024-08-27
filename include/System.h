
#include <string>
#include <thread>
#include <opencv2/core/core.hpp>

#include "DBoW3.h"


namespace EASY_SLAM{

class System
{

public:
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2
    };

public:

    System(const std::string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true);
    // cv::Mat TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp);

private:

    eSensor mSensor;
    // for ORB+DBoW3
    DBoW3::Vocabulary* mpORBvoc;
    DBoW3::Database* mpDbowdb;
    // for deep features



}; // class System

} // namespace EASY_SLAM