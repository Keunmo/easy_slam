#include <vector>
#include <opencv2/opencv.hpp>


namespace EASY_SLAM{

    // void ComputeHomography(const std::vector<cv::Point2f> &src_points,
    //                     const std::vector<cv::Point2f> &dst_points,
    //                     cv::Mat &homography) {
    //     homography = cv::findHomography(src_points, dst_points, cv::RANSAC);
    // }

    // void ComputeFundamental(const std::vector<cv::Point2f> &src_points,
    //                         const std::vector<cv::Point2f> &dst_points,
    //                         cv::Mat &fundamental) {
    //     fundamental = cv::findFundamentalMat(src_points, dst_points, cv::FM_RANSAC);
    // }

    // void ComputeEssential(const cv::Mat &fundamental, const cv::Mat &intrinsics, cv::Mat &essential) {
    //     essential = intrinsics.t() * fundamental * intrinsics;
    // }

    // void solvePnP(const std::vector<cv::Point3f> &pts_3d,
    //             const std::vector<cv::Point2f> &pts_2d,
    //             const cv::Mat &intrinsics,
    //             const cv::Mat &dist_coeffs,
    //             cv::Mat &rvec,
    //             cv::Mat &tvec,
    //             bool use_extrinsic_guess) {
    //     cv::solvePnP(pts_3d, pts_2d, intrinsics, dist_coeffs, rvec, tvec, use_extrinsic_guess);
    // }

} // namespace EASY_SLAM