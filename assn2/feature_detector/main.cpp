#include <iostream>
#include <utility>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

const int surf_threshold = 5000;
const int orb_threshold = 15000;
const float good_match_percent = 0.05f;
enum detector_type { SURF_DETECTOR, ORB_DETECTOR};

void store_image(Mat img, string img_name)
{
	imwrite(img_name, img);
}

// SURF: 
// +) 3 times faster than SIFT
// +) good at rotation and blurring
// -) not good at illumination,
//    view point
//    returns descriptor and key point vector
pair<Mat, vector<KeyPoint>> surf(Mat src_img, int frame_id)
{
	// detecting the features
	Ptr<SURF> detector = SURF::create( surf_threshold );
	vector<KeyPoint> keypoints;
	Mat descriptor;
	detector->detectAndCompute(src_img, noArray(), keypoints, descriptor);
	
	// writing the image
	Mat marked_image;
	drawKeypoints(src_img, keypoints, marked_image);
	store_image(marked_image, to_string(frame_id)+".jpg");
	
	// vector of key points
	return make_pair(descriptor, keypoints);
}

// ORB:
// +) rotation, scale invariant
// +) better than SURF
// +) good for panorama stitching
// returns descriptor and keypoints of the image
pair<Mat, vector<KeyPoint>> orb(Mat src_image, int frame_id)
{
	vector<KeyPoint> src_key_points;
	Mat descriptor;
	// detecting features and computing descriptors
	Ptr<Feature2D> orb_detector = ORB::create(orb_threshold);
	orb_detector->detectAndCompute(src_image, Mat(), src_key_points, descriptor);

	// writing the image
	Mat marked_image;
	drawKeypoints(src_image, src_key_points, marked_image);
	store_image(marked_image, to_string(frame_id)+".jpg");
	
	return make_pair(descriptor, src_key_points);
}

// this function reads all images from folder path given in the arg
// and returns  vector of images, vector of descriptors and 2D vector whose (i,j) th element represents
// jth feature of ith image
pair< vector<Mat>, pair< vector<Mat>, vector<vector<KeyPoint>>> > get_desc_kpoint_image(string folder_path, detector_type detector)
{
	vector<vector<KeyPoint>> image_key_points;
	vector<Mat> descriptors, images;
	// complete path of the images
	vector<cv::String> img_file_names;
	// reading the file names
	glob(folder_path + "/*.jpg", img_file_names, false);
	int images_size = img_file_names.size();
	// traversing the images
	for(int i=0; i<images_size; i++)
	{
		// reading the files
		Mat src_image = imread(img_file_names[i], IMREAD_GRAYSCALE);
		if(src_image.empty())	cout << "Can not read image...\n";
		
		pair<Mat, vector<KeyPoint>> des_kp;
	
		// type of detector to use	
		switch(detector)
		{
			case SURF_DETECTOR:
				des_kp = surf(src_image, i);
				break;
			case ORB_DETECTOR:
				des_kp = orb(src_image, i);
				break;
		}

		images.push_back(src_image);
		descriptors.push_back(des_kp.first);
		image_key_points.push_back( des_kp.second );	
	}

	return make_pair(images, make_pair(descriptors, image_key_points));
}

// this function returns vector of pairs. Each pair is made up of two vectors of point. Point in one vector matches with corresponding point (same index) of the other vector.
// first vector is for first image, second for second image.
pair<vector<Point2f>, vector<Point2f>> get_matching_points(	Mat &img1, Mat &img2, Mat &descriptor1, Mat &descriptor2, 
								vector<KeyPoint> &key_point1, vector<KeyPoint> &key_point2, 
								int &frame_id1, int &frame_id2)
{
	// matching the features
	vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptor1, descriptor2, matches, Mat());
	
	// sorting the matches by their scores
	sort(matches.begin(), matches.end());

	// removing not so good mathces
	const int num_of_good_matches = matches.size()*good_match_percent;
	matches.erase(matches.begin()+num_of_good_matches, matches.end());

	// drawing the top matches and storing it in current dir
	Mat image_matches;
	drawMatches(img1, key_point1, img2, key_point2, matches, image_matches);
	store_image(image_matches, to_string(frame_id1)+"_"+to_string(frame_id2)+".jpg");

	// extracting the location of good matching points from two images
	vector<Point2f> points1, points2;
	for(int i=0; i<num_of_good_matches; i++)
	{
		points1.push_back( key_point1[ matches[i].queryIdx ].pt );
		points2.push_back( key_point2[ matches[i].trainIdx ].pt );
	}

	return make_pair(points1, points2);
}

int main(int argc, char* argv[])
{
	// getting the image folder path
	string folder_path;
	cout << "Enter path to folder:\n" << 
		"default path is: /home/dwijesh/Documents/sem5/vision/assns/assn2_data/1" 
		<< endl;
	cin >> folder_path;
	// getting image matrix, descriptors and key points for all images in given folder
	pair< vector<Mat>, pair< vector<Mat>, vector<vector<KeyPoint>>> > all_mappings = get_desc_kpoint_image(folder_path, ORB_DETECTOR);

	int frame_id1=1, frame_id2 = 2;
	pair<vector<Point2f>, vector<Point2f>> matching_points = get_matching_points( 	(all_mappings.first)[1], (all_mappings.first)[2], ((all_mappings.second).first)[1], 
												((all_mappings.second).first)[2], ((all_mappings.second).second)[1],
												((all_mappings.second).second)[2], frame_id1, frame_id2);

	return 0;
}
