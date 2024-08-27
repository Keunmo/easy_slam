#include "Frame.h"

#include <ext/alloc_traits.h>
#include <ext/new_allocator.h>
#include <limits.h>
#include <math.h>
#include <opencv2/calib3d.hpp>
#include <thread>
#include <algorithm>
#include <memory>
#include <utility>

#include "FeatureExtractor.h"
#include "MapPoint.h"

namespace EASY_SLAM
{

long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;

Frame::Frame()
{}

// Copy constructor
Frame::Frame(const Frame &frame)
    :mpORBvoc(frame.mpORBvoc), mpLocFeatExtractorLeft(frame.mpLocFeatExtractorLeft), mpLocFeatExtractorRight(frame.mpLocFeatExtractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()), mbf(frame.mbf), mThDepth(frame.mThDepth), 
     N(frame.N), mvKeys(frame.mvKeys), mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight), 
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mDescriptors(frame.mDescriptors.clone()), 
     mDescriptorsRight(frame.mDescriptorsRight.clone()), mGlobalDescriptor(frame.mGlobalDescriptor.clone()), mvpMapPoints(frame.mvpMapPoints), 
     mvbOutlier(frame.mvbOutlier), mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF)
{
    if (!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

// Stereo constructor, use deep global feature.
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &TimeStamp, FeatureExtractor* locFeatExtractorLeft, FeatureExtractor* locFeatExtractorRight, DBoW3::Vocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvoc(voc), mpLocFeatExtractorLeft(locFeatExtractorLeft), mpLocFeatExtractorRight(locFeatExtractorRight), mTimeStamp(TimeStamp), 
     mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mpReferenceKF(static_cast<KeyFrame*>(NULL)) 
{
    // Frame ID
    mnId = nNextId++;

    std::thread threadLocLeft(&Frame::ExtractFeature, this, 0, imLeft);
    std::thread threadLocRight(&Frame::ExtractFeature, this, 1, imRight);
    threadLocLeft.join();
    threadLocRight.join();

    N = mvKeys.size();
    
    if(mvKeys.empty())
        return;
    
    UndistortKeyPoints();
    ComputeStereoMatches();

    mvpMapPoints = std::vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));
    mvbOutlier = std::vector<bool>(N, false);

    if (mbInitialComputations){
        // ComputeImageBounds(imLeft);
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations = false;
    }
    mb = mbf/fx;
}

void Frame::ExtractFeature(int flag, const cv::Mat &im){
    if (flag == 0){
        mpLocFeatExtractorLeft->Extract(im, std::to_string(mnId), false, mvKeys, mDescriptors);
    }
    else {
        mpLocFeatExtractorRight->Extract(im, std::to_string(mnId), false, mvKeysRight, mDescriptorsRight);
    }
}

void Frame::SetPose(cv::Mat Tcw){
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices(){
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRwc*mtcw;
}

void Frame::ComputeBoW(){
    if (mBowVec.empty()){
        std::vector<cv::Mat> vCurrentDesc;
        vCurrentDesc.reserve(mDescriptors.rows);
        for(int i=0; i<mDescriptors.rows; i++)
            vCurrentDesc.emplace_back(mDescriptors.row(i));
        mpORBvoc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

void Frame::UndistortKeyPoints(){
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches(){
    mvuRight = std::vector<float>(N, -1.0f);
    mvDepth = std::vector<float>(N, -1.0f);

    const int thOrbDist

}



}   // namespace EASY_SLAM