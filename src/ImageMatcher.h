#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "./lib/AKAZE.h"

// Image matching options
const float MIN_H_ERROR = 2.50f;            ///< Maximum error in pixels to accept an inlier
const float DRATIO = 0.80f;                 ///< NNDR Matching value

static bool checkHomography(cv::Mat& h) {
    //find the determinant, if < 0, orientation non-preserving
    const double det = h.at<double>(0, 0) * h.at<double>(1, 1) - h.at<double>(1, 0) * h.at<double>(1, 0);
    if (det < 0)
        return false;
    const double N1 = sqrt(h.at<double>(0, 0) * h.at<double>(0, 0) + h.at<double>(1, 0) * h.at<double>(1, 0));
    if (N1 > 4 || N1 < 0.1)
        return false;

    const double N2 = sqrt(h.at<double>(0, 1) * h.at<double>(0, 1) + h.at<double>(1, 1) * h.at<double>(1, 1));
    if (N2 > 4 || N2 < 0.1)
        return false;

    const double N3 = sqrt(h.at<double>(2, 0) * h.at<double>(2, 0) + h.at<double>(2, 1) * h.at<double>(2, 1));
    if (N3 > 0.002)
        return false;

    return true;
}

static double det(double x1, double y1, double x2, double y2) {
    return (x1 * y2) - (y1 * x2);
}

class AKazeDetector {
public:
    void detect(cv::Mat& img1, cv::Mat& img2, std::vector<cv::KeyPoint>& kpts1, std::vector<cv::KeyPoint>& kpts2,
            cv::Mat& desc1, cv::Mat& desc2) {

        cv::Mat img1_32, img2_32;

        // Convert the images to float
        img1.convertTo(img1_32, CV_32F, 1.0/255.0, 0);
        img2.convertTo(img2_32, CV_32F, 1.0/255.0, 0);

        //for now we hard code the kaze options
        AKAZEOptions options;

        // Create the first AKAZE object
        options.img_width = img1.cols;
        options.img_height = img1.rows;
        libAKAZE::AKAZE evolution1(options);

        // Create the second AKAZE object
        options.img_width = img2.cols;
        options.img_height = img2.rows;
        libAKAZE::AKAZE evolution2(options);

        evolution1.Create_Nonlinear_Scale_Space(img1_32);
        evolution1.Feature_Detection(kpts1);
        evolution1.Compute_Descriptors(kpts1, desc1);

        evolution2.Create_Nonlinear_Scale_Space(img2_32);
        evolution2.Feature_Detection(kpts2);
        evolution2.Compute_Descriptors(kpts2, desc2);
    }
    std::string name() {
        return "akaze";
    }
private:
};

class SurfDetector {
public:
    void detect(cv::Mat& img1, cv::Mat& img2, std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
            cv::Mat& desc1, cv::Mat& desc2) {
        cv::SURF surf(500, 4, 2, false, false);
        surf(img1, cv::Mat(), kp1, desc1);
        surf(img2, cv::Mat(), kp2, desc2);
    }
    std::string name() {
        return "surf";
    }

private:

};

struct BruteForceType {
    static std::string name;
};
struct BruteForceHammingType {
    static std::string name;
};
struct BruteForceL1Type {
    static std::string name;
};
struct BruteForceHamming2Type {
    static std::string name;
};
struct FlannBasedType {
    static std::string name;
};

std::string BruteForceType::name = "BruteForce";
std::string BruteForceHammingType::name = "BruteForce-Hamming";
std::string BruteForceL1Type::name = "BruteForce-L1";
std::string BruteForceHamming2Type::name = "BruteForceHamming(2)";
std::string FlannBasedType::name = "FlannBased";

/*!
@class ImageMatcher
@brief Matches 2 images
 */
template <typename FeatureDetector, typename DescriptorMatcherType>
class ImageMatcher {
public:
    ImageMatcher(cv::Mat& img1, cv::Mat& img2) {
        m_img1 = img1.clone();
        m_img2 = img2.clone();
    }
    void match() {

        double t1 = 0.0, t2 = 0.0;
        double t_detect = 0.0, t_match = 0.0, t_homography = 0.0;
        t1 = cv::getTickCount();
        m_feature_detector.detect(m_img1, m_img2, kpts1, kpts2, desc1, desc2);
        t2 = cv::getTickCount();
        t_detect = 1000.0*(t2 - t1)/cv::getTickFrequency();

        t1 = cv::getTickCount();
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(DescriptorMatcherType::name);
        matcher->knnMatch(desc1, desc2, dmatches, 2);

        t2 = cv::getTickCount();
        t_match = 1000.0*(t2 - t1)/ cv::getTickFrequency();

        matches2points_nndr(kpts1, kpts2, dmatches, matches, DRATIO);

        t1 = cv::getTickCount();
        h = compute_inliers_ransac(matches, inliers, MIN_H_ERROR, false);
        t2 = cv::getTickCount();
        t_homography = 1000.0*(t2 - t1)/cv::getTickFrequency();

        // Compute the inliers statistics
        size_t nkpts1 = kpts1.size();
        size_t nkpts2 = kpts2.size();
        size_t nmatches = matches.size()/2;
        size_t ninliers = inliers.size()/2;
        size_t noutliers = nmatches - ninliers;
        float ratio = 100.0*((float) ninliers / (float) nmatches);

        // Show matching statistics
        std::cout << "Number of Keypoints Image 1: " << nkpts1 << std::endl;
        std::cout << "Number of Keypoints Image 2: " << nkpts2 << std::endl;
        std::cout << "Features Extraction Time (ms): " << t_detect << std::endl;
        std::cout << "Matching Descriptors Time (ms): " << t_match << std::endl;
        std::cout << "Homography Time (ms): " << t_homography << std::endl;
        std::cout << "Number of Matches: " << nmatches << std::endl;
        std::cout << "Number of Inliers: " << ninliers << std::endl;
        std::cout << "Number of Outliers: " << noutliers << std::endl;
        std::cout << "Inliers Ratio: " << ratio << std::endl << std::endl;

        //print out the fundemental matrix
        for(int j = 0; j < h.rows; ++j) {
            for(int i = 0; i < h.cols; ++i) {
                std::cout << h.at<double>(j, i) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Valid Homography: " << checkHomography(h) << std::endl;
    }
    void show() {

        //convert the image to 3 Channels
        cv::Mat img1_rgb = cv::Mat(cv::Size(m_img1.cols, m_img1.rows), CV_8UC3);
        cv::Mat img2_rgb = cv::Mat(cv::Size(m_img2.cols, m_img2.rows), CV_8UC3);

        // Prepare the visualization
        cvtColor(m_img1, img1_rgb, cv::COLOR_GRAY2BGR);
        cvtColor(m_img2, img2_rgb, cv::COLOR_GRAY2BGR);

        //draw the results, combine the images
        int combined_width = img1_rgb.cols + img2_rgb.cols;
        int combined_height = cv::max(img1_rgb.rows, img2_rgb.rows);

        cv::Mat combined(combined_height, combined_width, CV_8UC3);

        //cv::drawKeypoints(img1_rgb, kpts1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        //cv::drawKeypoints(img2_rgb, kpts2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(img1_rgb, kpts1, img1_rgb, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(img2_rgb, kpts2, img2_rgb, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        //draw_keypoints(img1_rgb, kpts1);
        //draw_keypoints(img2_rgb, kpts2);

        //construct the combined image
        for(int y=0; y < img1_rgb.rows; y++) {
            for(int x=0; x < img1_rgb.cols; x++) {
                combined.at<cv::Vec3b>(y,x)[0] = img1_rgb.at<cv::Vec3b>(y,x)[0];
                combined.at<cv::Vec3b>(y,x)[1] = img1_rgb.at<cv::Vec3b>(y,x)[1];
                combined.at<cv::Vec3b>(y,x)[2] = img1_rgb.at<cv::Vec3b>(y,x)[2];
            }
        }

        for(int y=0; y < img2_rgb.rows; y++) {
            for(int x=0; x < img2_rgb.cols; x++) {
                combined.at<cv::Vec3b>(y,img1_rgb.cols + x)[0] = img2_rgb.at<cv::Vec3b>(y,x)[0];
                combined.at<cv::Vec3b>(y,img1_rgb.cols + x)[1] = img2_rgb.at<cv::Vec3b>(y,x)[1];
                combined.at<cv::Vec3b>(y,img1_rgb.cols + x)[2] = img2_rgb.at<cv::Vec3b>(y,x)[2];
            }
        }

        //draw the inliers
        for(unsigned int i=0; i < inliers.size(); i+=2) {
            cv::line(combined, cv::Point(inliers[i].x, inliers[i].y), cv::Point(img1_rgb.cols + inliers[i+1].x, inliers[i + 1].y), CV_RGB(0, 0, 255));
        }

        //wrap perspective for the original logo to the target
        std::vector<cv::Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0,0);
        obj_corners[1] = cvPoint( img1_rgb.cols, 0 );
        obj_corners[2] = cvPoint( img1_rgb.cols, img1_rgb.rows );
        obj_corners[3] = cvPoint( 0, img1_rgb.rows );
        std::vector<cv::Point2f> scene_corners(4);
        perspectiveTransform(obj_corners, scene_corners, h);

        //TODO: pick 4 inliers and try to warp perspective and verfiy cross ratio?

        double cross_ratio_orig = (det(obj_corners[0].x, obj_corners[0].y, obj_corners[3].x, obj_corners[3].y) *
                det(obj_corners[1].x, obj_corners[1].y, obj_corners[2].x, obj_corners[2].y))
                / (det(obj_corners[0].x, obj_corners[0].y, obj_corners[1].x, obj_corners[1].y) *
                det(obj_corners[3].x, obj_corners[3].y, obj_corners[2].x, obj_corners[2].y));

        double cross_ratio_transformed = (det(scene_corners[0].x, scene_corners[0].y, scene_corners[3].x, scene_corners[3].y) *
                det(scene_corners[1].x, scene_corners[1].y, scene_corners[2].x, scene_corners[2].y))
                / (det(scene_corners[0].x, scene_corners[0].y, scene_corners[1].x, scene_corners[1].y) *
                det(scene_corners[3].x, scene_corners[3].y, scene_corners[2].x, scene_corners[2].y));

        double original_area = cv::contourArea(obj_corners);
        double area = cv::contourArea(scene_corners);
        double a = det(obj_corners[0].x, obj_corners[0].y, obj_corners[3].x, obj_corners[3].y);
        double b = det(obj_corners[1].x, obj_corners[1].y, obj_corners[2].x, obj_corners[2].y);
        double c = det(obj_corners[0].x, obj_corners[0].y, obj_corners[1].x, obj_corners[1].y);
        double d = det(obj_corners[3].x, obj_corners[3].y, obj_corners[2].x, obj_corners[2].y);

        std::cout << "a: " << a << " b: " << b << " c: " << c << " d: " << d << std::endl;
        std::cout << "Cross Ratio Original" << cross_ratio_orig << std::endl;
        std::cout << "Cross Ratio Transformed" << cross_ratio_transformed << std::endl;
        std::cout << "Original Area: " << original_area << std::endl;
        std::cout << "Transformed Area: " << area << std::endl;

        //draw the perspective transform
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        cv::line( combined, scene_corners[0] + cv::Point2f(img1_rgb.cols, 0), scene_corners[1] + cv::Point2f(img1_rgb.cols, 0), cv::Scalar(255, 0, 0), 4 );
        cv::line( combined, scene_corners[1] + cv::Point2f(img1_rgb.cols, 0), scene_corners[2] + cv::Point2f(img1_rgb.cols, 0), cv::Scalar(255, 0, 0), 4 );
        cv::line( combined, scene_corners[2] + cv::Point2f(img1_rgb.cols, 0), scene_corners[3] + cv::Point2f(img1_rgb.cols, 0), cv::Scalar(255, 0, 0), 4 );
        cv::line( combined, scene_corners[3] + cv::Point2f(img1_rgb.cols, 0), scene_corners[0] + cv::Point2f(img1_rgb.cols, 0), cv::Scalar(255, 0, 0), 4 );

        //cv::imshow("img 1", img1_rgb);
        //cv::imshow("img 2", img2_rgb);
        //resize the results
        double scale_ratio = 1080.0 / combined.cols;
        cv::resize(combined, combined, cvSize(combined.cols * scale_ratio, combined.rows * scale_ratio));
        cv::imshow(m_feature_detector.name(), combined);
        cv::waitKey(0);
    }

private:
    cv::Mat m_img1, m_img2;
    std::vector<cv::KeyPoint> kpts1, kpts2;
    std::vector<cv::Point2f> matches, inliers;
    std::vector<std::vector<cv::DMatch> > dmatches;
    cv::Mat desc1, desc2;
    cv::Mat h;
    FeatureDetector m_feature_detector;
};