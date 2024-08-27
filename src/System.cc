#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <list>
#include <string>

#include "System.h"
#include "DBoW3.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"

namespace EASY_SLAM
{
System::System(const std::string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true)
:mSensor(sensor){
    std::cout << "Input sensor was set to: ";
    if(mSensor==MONOCULAR)
        std::cout << "Monocular" << std::endl;
    else if(mSensor==STEREO)
        std::cout << "Stereo" << std::endl;
    else if(mSensor==RGBD)
        std::cout << "RGB-D" << std::endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       std::cerr << "Failed to open settings file at: " << strSettingsFile << std::endl;
       exit(-1);
    }

    // Load voc file if gfeat is dbow3
    if(static_cast<std::string>(fsSettings["GlobalFeature"]) == "DBoW3"){
        std::string strVocFile = fsSettings["VocFile"];
        std::cout << "Loading vocabulary from: " << strVocFile << std::endl;
        mpORBvoc = new DBoW3::Vocabulary(strVocFile);
        mpDbowdb = new DBoW3::Database(*mpORBvoc, false, 0); // false = do not use direct index
    }
    // std::cout << mpDbowdb << std::endl;
    // load matcher model
    // if(static_cast<std::string>(fsSettings["Matcher"]) != "BF"){

    // }

    // start 



} // System

} // namespace EASY_SLAM
