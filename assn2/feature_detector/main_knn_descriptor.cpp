#include <iostream>
#include <utility>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

const int surf_threshold = 5000;
const int orb_threshold = 15000;
const float good_match_percent = 0.05f;
enum detector_type { SURF_DETECTOR, ORB_DETECTOR};
double persp_coor1 = INT_MAX;
double persp_coor2 = INT_MAX;
int height = 0;
int width = 0;
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
	store_image(marked_image, to_string(frame_id)+"_knn.jpg");
	
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
	vector<vector<DMatch>> knn_matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);
	
	// filtering matches with ratio test
	const float ratio_threshold = 0.9f;
	vector<DMatch> good_matches;
	for(size_t i=0; i< knn_matches.size(); i++)
		if(knn_matches[i][0].distance < ratio_threshold * knn_matches[i][1].distance)		good_matches.push_back(knn_matches[i][0]);
		else if(knn_matches[i][1].distance < ratio_threshold * knn_matches[i][0].distance)	good_matches.push_back(knn_matches[i][1]);

	// drawing matches
	Mat img_matches;
	drawMatches(img1, key_point1, img2, key_point2, good_matches, img_matches, Scalar::all(-1),  Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	store_image(img_matches, to_string(frame_id1)+"_"+to_string(frame_id2)+"_knn.jpg");

	// extracting the location of good matching points from two images
	vector<Point2f> points1, points2;
	int num_of_good_matches = good_matches.size();
	for(int i=0; i<num_of_good_matches; i++)
	{
		points1.push_back( key_point1[ good_matches[i].queryIdx ].pt );
		points2.push_back( key_point2[ good_matches[i].trainIdx ].pt );
	}

	return make_pair(points1, points2);
}

// input:	parameter is complete information.    < vector of all images, < corresponding vector of descriptors, vector of key points for each of the images >>
// returns: 	an upper triangle of a 2D matrix. aij th element is: mapping of corresponding points from image i to image j. A pair of two vectors.
// 		kth element of first vector is mapped kth element of second vector.
vector<vector< pair< vector<Point2f>, vector<Point2f>> >> get_pair_wise_matching_points(pair< vector<Mat>, pair< vector<Mat>, vector<vector<KeyPoint>>> > &all_mappings)
{
	vector<vector< pair< vector<Point2f>, vector<Point2f>> >>	pair_wise_matching_points;
	int image_count = (all_mappings.first).size();
	// getting pair wise matching points
	for(int i=0; i<image_count; i++)
	{
		vector< pair< vector<Point2f>, vector<Point2f> > > pair_wise_matching_for_ith_image;
		for(int j=0; j<image_count; j++)
		{
			pair_wise_matching_for_ith_image.push_back( get_matching_points(
												(all_mappings.first)[i], (all_mappings.first)[j], ((all_mappings.second).first)[i], 
												((all_mappings.second).first)[j], ((all_mappings.second).second)[i],
												((all_mappings.second).second)[j], i, j
											)
								  );
		}
		pair_wise_matching_points.push_back(pair_wise_matching_for_ith_image);
	}
	return pair_wise_matching_points;
}

// input:	(pair of (vector of matching points)) for each pair of images
// returns:	2D matrix. aij == homography matrix for transformation from image i to image j
vector<vector< Mat >> get_pair_wise_homography_matrix(vector<vector< pair< vector<Point2f>, vector<Point2f>> >> &pair_wise_matching_points)
{
	vector<vector< Mat >> pair_wise_homography_matrix;
	int image_count = pair_wise_matching_points.size();
	// traversing all images
	for(int i=0; i<image_count; i++)
	{
		vector< Mat > hmatrix_imgi_wrt_others;
		// for a given image, find homography matrix wrt all other images
		for(int j=0; j<image_count ; j++){
			Mat hgraphy =  findHomography(pair_wise_matching_points[i][j].first, pair_wise_matching_points[i][j].second, RANSAC);
			if(i != j){
	            persp_coor1 = min(abs(hgraphy.at<double>(2,0)),persp_coor1);
				persp_coor2 = min(abs(hgraphy.at<double>(2,1)),persp_coor2);
			}
			hmatrix_imgi_wrt_others.push_back(hgraphy);
		}
		pair_wise_homography_matrix.push_back(hmatrix_imgi_wrt_others);
	}
	cout << persp_coor1 << " OKay " << persp_coor2 << endl;
	return pair_wise_homography_matrix;
}

bool scoring_condition1(Mat hmatrix){
	Mat point1 = (Mat_<double>(3,1) << 0,0,1);
	Mat point2 = (Mat_<double>(3,1) << width,0,1);	
	Mat point3 = (Mat_<double>(3,1) << width, height,1);
	Mat point4 = (Mat_<double>(3,1) << 0,height,1);	
	Mat new_point1 = hmatrix * point1;
	Mat new_point2 = hmatrix * point2;
	Mat new_point3 = hmatrix * point3;
	Mat new_point4 = hmatrix * point4;
	if(new_point1.at<double>(0,0) > new_point2.at<double>(0,0))
		return false;
	if(new_point2.at<double>(1,0) > new_point3.at<double>(1,0))
		return false;
	if(new_point4.at<double>(0,0) > new_point3.at<double>(0,0))
		return false;
	if(new_point1.at<double>(1,0) > new_point4.at<double>(1,0))
		return false;
	return true;
}

double scoring_condition2(Mat hmatrix)
{
	Mat point1 = (Mat_<double>(3,1) << 0,0,1);
	Mat point2 = (Mat_<double>(3,1) << width,0,1);	
	Mat point3 = (Mat_<double>(3,1) << width, height,1);
	Mat point4 = (Mat_<double>(3,1) << 0,height,1);	
	Mat new_point1 = hmatrix * point1;
	Mat new_point2 = hmatrix * point2;
	Mat new_point3 = hmatrix * point3;
	Mat new_point4 = hmatrix * point4;
	
	vector<Point2f> poly1, poly2;
	poly1.push_back(Point2f(point1.at<double>(0, 0), point1.at<double>(1, 0)));
	poly1.push_back(Point2f(point2.at<double>(0, 0), point2.at<double>(1, 0)));
	poly1.push_back(Point2f(point3.at<double>(0, 0), point3.at<double>(1, 0)));
	poly1.push_back(Point2f(point4.at<double>(0, 0), point4.at<double>(1, 0)));

	poly2.push_back(Point2f(new_point1.at<double>(0, 0), new_point1.at<double>(1, 0)));
	poly2.push_back(Point2f(new_point2.at<double>(0, 0), new_point2.at<double>(1, 0)));
	poly2.push_back(Point2f(new_point3.at<double>(0, 0), new_point3.at<double>(1, 0)));
	poly2.push_back(Point2f(new_point4.at<double>(0, 0), new_point4.at<double>(1, 0)));

	double area1 = contourArea(poly1), area2 = contourArea(poly2);
	return (area1/area2);
}

double score(Mat hmatrix){
	double coor1 = hmatrix.at<double>(2,0);
	double coor2 = hmatrix.at<double>(2,1);
	double score = 0;
	if((abs(coor1)/persp_coor1 < 100 && abs(coor1)/persp_coor1 >1) || (abs(coor2)/persp_coor2 < 100 && abs(coor2)/persp_coor2 >1)){
		if(scoring_condition1(hmatrix)){
			double ratio = scoring_condition2(hmatrix);
			double power = (1-ratio) * (1-ratio);
			score = 1/(exp(power));
			cout << "score is: " << score << endl;
		}
	}
	return score;
}

// sort in decreasing order wrt first element:score of the pair
bool sort_des(const pair<double, pair<int, int>> &homography1, const pair<double, pair<int, int>> &homography2)
{
	return homography1.first > homography2.first;
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
	pair< vector<Mat>, pair< vector<Mat>, vector<vector<KeyPoint>>> > all_mappings = get_desc_kpoint_image(folder_path, SURF_DETECTOR);
	height = (all_mappings.first[0]).rows;
	width = (all_mappings.first[0]).cols;
	int image_count = all_mappings.first.size();
	// getting matching points for all possible pairs of images
	vector<vector< pair< vector<Point2f>, vector<Point2f>> >> pair_wise_matching_points = get_pair_wise_matching_points(all_mappings);

	// getting homography matrix for all possible pairs of images
	vector<vector< Mat >> pair_wise_homography_matrix = get_pair_wise_homography_matrix(pair_wise_matching_points);
	vector< pair< double, pair<int, int> >> good_homography_indices;

	// storing homographies with their scores
	for(int i=0; i<image_count; i++)
		for(int j=0; j<image_count; j++)
			good_homography_indices.push_back( make_pair( score(pair_wise_homography_matrix[i][j]), make_pair(i, j) ) );

	// sorting homographies wrt their scores and selecting n-1 best homographihes
	sort(good_homography_indices.begin(), good_homography_indices.end(), sort_des);
	
	// before erasing
	cout << "before erasing after sorting\n";
	for(int i=0; i<good_homography_indices.size(); i++)	cout << good_homography_indices[i].first << ": " <<
								good_homography_indices[i].second.first << " -> " << 
								good_homography_indices[i].second.second << endl;
	// selecting n-1 entries
	good_homography_indices.erase( good_homography_indices.begin()+image_count-1, good_homography_indices.end());

	// printing n-1 sorted indices of homography
	cout << "after erasing\n";
	for(int i=0; i<image_count-1; i++)	cout << good_homography_indices[i].first << ": " <<
							good_homography_indices[i].second.first << " -> " << 
							good_homography_indices[i].second.second << endl;
	return 0;
}
