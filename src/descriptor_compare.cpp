//=============================================================================
//
// akaze_match.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Toshiba Research Europe Ltd (1)
//               TrueVision Solutions (2)
// Date: 07/10/2014
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2014, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
* @file akaze_match.cpp
* @brief Main program for matching two images with AKAZE features
* @date Oct 07, 2014
* @author Pablo F. Alcantarilla
*/

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "./lib/AKAZE.h"
#include "ImageMatcher.h"

using namespace std;

/**
* @brief This function parses the command line arguments for setting A-KAZE parameters
* and image matching between two input images
* @param options Structure that contains A-KAZE settings
* @param img_path1 Path for the first input image
* @param img_path2 Path for the second input image
* @param homography_path Path for the file that contains the ground truth homography
*/
int parse_input_options(AKAZEOptions &options, std::string &img_path1,
        std::string &img_path2, std::string &homography_path,
        int argc, char *argv[]);

bool isLandscape(const cv::Mat &image) {
    return image.cols > image.rows;
}

cv::Mat &resizeToFit(cv::Mat &image, CvSize size) {
    //if it is a smaller image, no resize is needed
    if (image.rows < size.height && image.cols < size.width) {
        return image;
    }
    else {
        //determine which side to fit the scale
        double scale_ratio = 1.0;
        if (image.cols > image.rows)
            scale_ratio = (double) size.width / (double) image.cols;
        else {
            scale_ratio = (double) size.height / (double) image.rows;
        }
        cv::resize(image, image, cvSize(image.cols * scale_ratio, image.rows * scale_ratio));
    }
    return image;
}

/* ************************************************************************* */
int main(int argc, char *argv[]) {

    // Variables
    AKAZEOptions options;
    cv::Mat img1, img2;
    string img_path1, img_path2, homography_path;

    // Parse the input command line options
    if (parse_input_options(options, img_path1, img_path2, homography_path, argc, argv))
        return -1;

    // Read image 1 and if necessary convert to grayscale
    img1 = cv::imread(img_path1, 0);
    if (img1.data == NULL) {
        cerr << "Error loading image 1: " << img_path1 << endl;
        return -1;
    }

    // Read image 2 and if necessary convert to grayscale.
    img2 = cv::imread(img_path2, 0);
    if (img2.data == NULL) {
        cerr << "Error loading image 2: " << img_path2 << endl;
        return -1;
    }

    //do resizing
    if (isLandscape(img1))
        img1 = resizeToFit(img1, cvSize(640, 480));
    else
        img1 = resizeToFit(img1, cvSize(480, 640));

    if (isLandscape(img2))
        img2 = resizeToFit(img2, cvSize(640, 480));
    else
        img2 = resizeToFit(img2, cvSize(480, 640));

    ImageMatcher<AKazeDetector, BruteForceHammingType> kaze_matcher(img1, img2);
    ImageMatchResult result_kaze = kaze_matcher.match();
    kaze_matcher.show();
    std::cout << "akaze" << std::endl;
    std::cout << result_kaze << std::endl;

    ImageMatcher<SurfDetector, BruteForceType> surf_matcher(img1, img2);
    ImageMatchResult result_surf = surf_matcher.match();
    surf_matcher.show();
    std::cout << "surf" << std::endl;
    std::cout << result_surf << std::endl;

    ImageMatcher<GPUSurfDetector, BruteForceType> gpu_surf_matcher(img1, img2);
    ImageMatchResult result_gpu_surf = gpu_surf_matcher.match();
    gpu_surf_matcher.show();
    std::cout << "gpu surf" << std::endl;
    std::cout << result_gpu_surf << std::endl;
}

/* ************************************************************************* */
int parse_input_options(AKAZEOptions &options, std::string &img_path1, std::string &img_path2,
        std::string &homography_path, int argc, char *argv[]) {

    // If there is only one argument return
    if (argc == 1) {
        show_input_options_help(1);
        return -1;
    }
        // Set the options from the command line
    else if (argc >= 2) {

        // Load the default options
        options = AKAZEOptions();

        if (!strcmp(argv[1], "--help")) {
            show_input_options_help(1);
            return -1;
        }

        img_path1 = argv[1];
        img_path2 = argv[2];

        if (argc >= 4)
            homography_path = argv[3];

        for (int i = 3; i < argc; i++) {
            if (!strcmp(argv[i], "--soffset")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    options.soffset = atof(argv[i]);
                }
            }
            else if (!strcmp(argv[i], "--omax")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    options.omax = atof(argv[i]);
                }
            }
            else if (!strcmp(argv[i], "--dthreshold")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    options.dthreshold = atof(argv[i]);
                }
            }
            else if (!strcmp(argv[i], "--sderivatives")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    options.sderivatives = atof(argv[i]);
                }
            }
            else if (!strcmp(argv[i], "--nsublevels")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    options.nsublevels = atoi(argv[i]);
                }
            }
            else if (!strcmp(argv[i], "--diffusivity")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    options.diffusivity = DIFFUSIVITY_TYPE(atoi(argv[i]));
                }
            }
            else if (!strcmp(argv[i], "--descriptor")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    options.descriptor = DESCRIPTOR_TYPE(atoi(argv[i]));

                    if (options.descriptor < 0 || options.descriptor > MLDB) {
                        options.descriptor = MLDB;
                    }
                }
            }
            else if (!strcmp(argv[i], "--descriptor_channels")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    options.descriptor_channels = atoi(argv[i]);

                    if (options.descriptor_channels <= 0 || options.descriptor_channels > 3) {
                        options.descriptor_channels = 3;
                    }
                }
            }
            else if (!strcmp(argv[i], "--descriptor_size")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                }
                else {
                    options.descriptor_size = atoi(argv[i]);

                    if (options.descriptor_size < 0) {
                        options.descriptor_size = 0;
                    }
                }
            }
            else if (!strcmp(argv[i], "--verbose")) {
                options.verbosity = true;
            }
            else if (!strncmp(argv[i], "--", 2))
                cerr << "Unknown command " << argv[i] << endl;
        }
    }
    else {
        cerr << "Error introducing input options!!" << endl;
        show_input_options_help(1);
        return -1;
    }

    return 0;
}
