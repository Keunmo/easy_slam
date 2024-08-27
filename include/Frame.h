#ifndef FRAME_H
#define FRAME_H

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <cstddef>

#include "MapPoint.h"
#include "DBoW3.h"
#include "FeatureExtractor.h"
#include "KeyFrame.h"

namespace EASY_SLAM { class FeatureExtractor; }

namespace EASY_SLAM
{

class MapPoint;
class KeyFrame;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, FeatureExtractor* mpLocFeatExtractorLeft, FeatureExtractor* mpLocFeatExtractorRight, DBoW3::Vocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for Monocular cameras.
    // Frame(const cv::Mat &imGray, const double &timeStamp, FeatureExtractor* extractor, DBoW3::Vocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Extract Feature on the image. 0 for left image and 1 for right image, 2 for global feature.
    void ExtractFeature(int flag, const cv::Mat &im);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose.
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }

public:
    // Vocabulary used for relocalization.
    DBoW3::Vocabulary *mpORBvoc;

    // Feature extractor. The right is used only in the stereo case.
    FeatureExtractor *mpLocFeatExtractorLeft;
    FeatureExtractor *mpLocFeatExtractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // Number of KeyPoints.
    int N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    DBoW3::BowVector mBowVec;
    DBoW3::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors, mDescriptorsRight;

    // Grobal descriptor.
    cv::Mat mGlobalDescriptor;

    // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;

    // Camera pose.
    cv::Mat mTcw;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;

    // // Scale pyramid info.
    // int mnScaleLevels;
    // float mfScaleFactor;
    // float mfLogScaleFactor;
    // vector<float> mvScaleFactors;
    // vector<float> mvInvScaleFactors;
    // vector<float> mvLevelSigma2;
    // vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once). no. stereo dont need it.
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;

private:
    // // Undistort keypoints given OpenCV distortion parameters.
    // // Only for the RGB-D case. Stereo must be already rectified!
    // // (called in the constructor). no. stereo dont need it.
    // void UndistortKeyPoints();

    // // Computes image bounds for the undistorted image (called in the constructor).
    // void ComputeImageBounds(const cv::Mat &imLeft);

    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw; //==mtwc

}; // class Frame

} // namespace EASY_SLAM