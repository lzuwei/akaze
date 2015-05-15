#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include "./lib/AKAZE.h"

// Image matching options
const float MIN_H_ERROR = 2.50f;            ///< Maximum error in pixels to accept an inlier
const float DRATIO = 0.80f;                 ///< NNDR Matching value
const int MAX_NUM_LIMITS = 10;

struct RansacParams {
    float ransac_epsilon;
    int ransac_iterations;
    //
    float verify_points_epsilon;
    float verify_descr_epsilon;
    //
    uint32_t num_limits;
    float coefs[MAX_NUM_LIMITS];
    //
    int max_angle;
    int limit0;
    int limit300;
    // RTM
    float coefs2[MAX_NUM_LIMITS];

    RansacParams() :
            ransac_epsilon(5.0),
            ransac_iterations(1000),
            verify_points_epsilon(15.0),
            verify_descr_epsilon(2.0),
            num_limits(5),
            max_angle(32),
            limit0(6),
            limit300(30){
        coefs[0] = 85.0;
        coefs[1] = 88.0;
        coefs[2] = 90.0;
        coefs[3] = 92.0;
        coefs[4] = 94.0;
    }
};

class ImageMatchResult {
public:
    ImageMatchResult(int nkpts1, int nkpts2, int nmatches, int ninliers, int noutliers, double ratio,
        bool valid_homography, double t_detect, double t_match, double t_homography) :
        nkpts1(nkpts1),
        nkpts2(nkpts2),
        nmatches(nmatches),
        ninliers(ninliers),
        noutliers(noutliers),
        ratio(ratio),
        valid_homography(valid_homography),
        t_detect(t_detect),
        t_match(t_match),
        t_homography(t_homography)
    {
    }
    friend std::ostream& operator <<(std::ostream& out, const ImageMatchResult& result);

    int nkpts1, nkpts2, nmatches, ninliers, noutliers;
    double ratio;
    bool valid_homography;
    double t_detect, t_match, t_homography;
};

std::ostream& operator <<(std::ostream& out, const ImageMatchResult& result) {
    out << "Number of Keypoints Image 1: " << result.nkpts1 << std::endl;
    out << "Number of Keypoints Image 2: " << result.nkpts2 << std::endl;
    out << "Features Extraction Time (ms): " << result.t_detect << std::endl;
    out << "Matching Descriptors Time (ms): " << result.t_match << std::endl;
    out << "Homography Time (ms): " << result.t_homography << std::endl;
    out << "Number of Matches: " << result.nmatches << std::endl;
    out << "Number of Inliers: " << result.ninliers << std::endl;
    out << "Number of Outliers: " << result.noutliers << std::endl;
    out << "Inliers Ratio: " << result.ratio << std::endl;
    out << "Valid Homography: " << result.valid_homography << std::endl;
    return out;
}

class AKazeDetector {
public:
    void detect(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> &kpts1, std::vector<cv::KeyPoint> &kpts2,
            cv::Mat &desc1, cv::Mat &desc2) {

        cv::Mat img1_32, img2_32;

        // Convert the images to float
        img1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
        img2.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

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
    void detect(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2,
            cv::Mat &desc1, cv::Mat &desc2) {
//        cv::SURF surf(500, 4, 2, false, false);
        cv::SURF surf(500, 3, 2, 0.006f, false);
        surf(img1, cv::Mat(), kp1, desc1);
        surf(img2, cv::Mat(), kp2, desc2);
    }

    std::string name() {
        return "surf";
    }

private:

};

class GPUSurfDetector {
public:
    void detect(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2,
            cv::Mat &desc1, cv::Mat &desc2) {

        cv::gpu::GpuMat kp1_gpu, kp2_gpu;
        cv::gpu::GpuMat desc1_gpu, desc2_gpu;
        std::vector<float> desc1_host;
        std::vector<float> desc2_host;

        cv::gpu::GpuMat img1_gpu(img1), img2_gpu(img2);
        cv::gpu::GpuMat gpu_image_mask;

        const bool extended_surf = false;
        cv::gpu::SURF_GPU surf(500, 3, 2, extended_surf, 0.006f, false);

        try {
            surf(img1_gpu, gpu_image_mask, kp1_gpu, desc1_gpu);
            surf(img2_gpu, gpu_image_mask, kp2_gpu, desc2_gpu);
        }
        catch (cv::Exception e) {
            std::cerr << "Exception Caught in " << e.file << " on line " << e.line << "." << std::endl;
            std::cerr << "Code: " << e.code << " Message: " << e.msg << std::endl;
            exit(1);
        }

        //if successful, download keypoints to host
        surf.downloadKeypoints(kp1_gpu, kp1);
        surf.downloadKeypoints(kp2_gpu, kp2);
        //surf.downloadDescriptors(desc1_gpu, desc1_host);
        //surf.downloadDescriptors(desc2_gpu, desc2_host);

        desc1 = cv::Mat(desc1_gpu);
        desc2 = cv::Mat(desc2_gpu);
    }

    std::string name() {
        return "gpu surf";
    }
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
template<typename FeatureDetector, typename DescriptorMatcherType>
class ImageMatcher {
public:
    ImageMatcher(cv::Mat &img1, cv::Mat &img2) {
        m_img1 = img1.clone();
        m_img2 = img2.clone();
    }

    ImageMatchResult match() {

        double t1 = 0.0, t2 = 0.0;
        double t_detect = 0.0, t_match = 0.0, t_homography = 0.0;
        t1 = cv::getTickCount();
        m_feature_detector.detect(m_img1, m_img2, kpts1, kpts2, desc1, desc2);
        t2 = cv::getTickCount();
        t_detect = 1000.0 * (t2 - t1) / cv::getTickFrequency();

        //Truncate the number of features extracted to 300
        if (desc1.rows > 300) {
            desc1 = desc1.rowRange(0, 300);
        }
        if (desc2.rows > 300) {
            desc2 = desc2.rowRange(0, 300);
        }
        t1 = cv::getTickCount();
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(DescriptorMatcherType::name);
        matcher->knnMatch(desc1, desc2, dmatches, 2);

        t2 = cv::getTickCount();
        t_match = 1000.0 * (t2 - t1) / cv::getTickFrequency();

        matches2points_nndr(kpts1, kpts2, dmatches, matches, DRATIO);

        t1 = cv::getTickCount();
        h = compute_inliers_ransac(matches, inliers, MIN_H_ERROR, false);
        t2 = cv::getTickCount();
        t_homography = 1000.0 * (t2 - t1) / cv::getTickFrequency();

        // Compute the inliers statistics
        size_t nkpts1 = kpts1.size();
        size_t nkpts2 = kpts2.size();
        size_t nmatches = matches.size() / 2;
        size_t ninliers = inliers.size() / 2;
        size_t noutliers = nmatches - ninliers;
        float ratio = 100.0 * ((float) ninliers / (float) nmatches);

        //print out the fundemental matrix
        for (int j = 0; j < h.rows; ++j) {
            for (int i = 0; i < h.cols; ++i) {
                std::cout << h.at<double>(j, i) << " ";
            }
            std::cout << std::endl;
        }

        std::vector<cv::Point2f> points(4);
        points[0] = cvPoint(0, 0);
        points[1] = cvPoint(m_img1.cols, 0);
        points[2] = cvPoint(m_img1.cols, m_img1.rows);
        points[3] = cvPoint(0, m_img1.rows);

        bool matched = checkHomography(h, points);
        return ImageMatchResult(nkpts1, nkpts2, nmatches, ninliers, noutliers, ratio, matched,
                t_detect, t_match, t_homography);
    }

    ImageMatchResult match2() {
        //implement visual system ransac and post search technique
        return ImageMatchResult(0,0,0,0,0,0.0,0,0.0,0.0,0.0);
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

        cv::drawKeypoints(img1_rgb, kpts1, img1_rgb, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(img2_rgb, kpts2, img2_rgb, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        //construct the combined image
        for (int y = 0; y < img1_rgb.rows; y++) {
            for (int x = 0; x < img1_rgb.cols; x++) {
                combined.at<cv::Vec3b>(y, x)[0] = img1_rgb.at<cv::Vec3b>(y, x)[0];
                combined.at<cv::Vec3b>(y, x)[1] = img1_rgb.at<cv::Vec3b>(y, x)[1];
                combined.at<cv::Vec3b>(y, x)[2] = img1_rgb.at<cv::Vec3b>(y, x)[2];
            }
        }

        for (int y = 0; y < img2_rgb.rows; y++) {
            for (int x = 0; x < img2_rgb.cols; x++) {
                combined.at<cv::Vec3b>(y, img1_rgb.cols + x)[0] = img2_rgb.at<cv::Vec3b>(y, x)[0];
                combined.at<cv::Vec3b>(y, img1_rgb.cols + x)[1] = img2_rgb.at<cv::Vec3b>(y, x)[1];
                combined.at<cv::Vec3b>(y, img1_rgb.cols + x)[2] = img2_rgb.at<cv::Vec3b>(y, x)[2];
            }
        }

        //draw the inliers
        for (unsigned int i = 0; i < inliers.size(); i += 2) {
            cv::line(combined, cv::Point(inliers[i].x, inliers[i].y), cv::Point(img1_rgb.cols + inliers[i + 1].x, inliers[i + 1].y), CV_RGB(0, 0, 255));
        }

        //wrap perspective for the original logo to the target
        std::vector<cv::Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0, 0);
        obj_corners[1] = cvPoint(img1_rgb.cols, 0);
        obj_corners[2] = cvPoint(img1_rgb.cols, img1_rgb.rows);
        obj_corners[3] = cvPoint(0, img1_rgb.rows);
        std::vector<cv::Point2f> scene_corners(4);
        cv::perspectiveTransform(obj_corners, scene_corners, h);

        //draw the perspective transform
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        cv::line(combined, scene_corners[0] + cv::Point2f(img1_rgb.cols, 0), scene_corners[1] + cv::Point2f(img1_rgb.cols, 0), cv::Scalar(255, 0, 0), 4);
        cv::line(combined, scene_corners[1] + cv::Point2f(img1_rgb.cols, 0), scene_corners[2] + cv::Point2f(img1_rgb.cols, 0), cv::Scalar(255, 0, 0), 4);
        cv::line(combined, scene_corners[2] + cv::Point2f(img1_rgb.cols, 0), scene_corners[3] + cv::Point2f(img1_rgb.cols, 0), cv::Scalar(255, 0, 0), 4);
        cv::line(combined, scene_corners[3] + cv::Point2f(img1_rgb.cols, 0), scene_corners[0] + cv::Point2f(img1_rgb.cols, 0), cv::Scalar(255, 0, 0), 4);

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