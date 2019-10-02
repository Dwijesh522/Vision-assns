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
enum alignment_type { HORIZONTAL, VERTICAL_FIRST, VERTICAL_SECOND, NONE};
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
	store_image(marked_image, to_string(frame_id)+".jpg");
	
	return make_pair(descriptor, src_key_points);
}

Mat equalize_hist(Mat img){
//Convert the image from BGR to YCrCb color space
	    Mat hist_equalized_image;
	    cvtColor(img, hist_equalized_image, COLOR_BGR2YCrCb);

	    //Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
	    vector<Mat> vec_channels;
	    split(hist_equalized_image, vec_channels); 

	    //Equalize the histogram of only the Y channel 
	    equalizeHist(vec_channels[0], vec_channels[0]);

	    //Merge 3 channels in the vector to form the color image in YCrCB color space.
	    merge(vec_channels, hist_equalized_image); 
		
	    //Convert the histogram equalized image from YCrCb to BGR color space again
	    cvtColor(hist_equalized_image,img, COLOR_YCrCb2BGR);
	    return img;
}
// this function returns colored images
vector<Mat> get_colored_images(string folder_path)
{
	vector<Mat> images;
	vector<cv::String> img_file_names;
	// reading the file names
	glob(folder_path + "/*.jpg", img_file_names, false);
	int images_size = img_file_names.size();
	// traversing the images
	for(int i=0; i<images_size; i++)
	{
		// reading the files
		Mat src_image_temp = imread(img_file_names[i], CV_LOAD_IMAGE_COLOR);
		if(src_image_temp.empty())	cout << "Can not read image...\n";
		
		Mat src_image ;
		Size dst_size = src_image.size();
		dst_size.width /= 2;
		dst_size.height /= 2;
		resize(src_image_temp, src_image, dst_size,0.5,0.5, CV_INTER_AREA);
		src_image = equalize_hist(src_image);
		images.push_back(src_image);
	}
	return images;
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
		Mat src_image_temp = imread(img_file_names[i], IMREAD_GRAYSCALE);
		if(src_image_temp.empty())	cout << "Can not read image...\n";
		
		Mat src_image ;
		Size dst_size = src_image.size();
		dst_size.width /= 2;
		dst_size.height /= 2;
		resize(src_image_temp, src_image, dst_size,0.5,0.5, CV_INTER_AREA);
		
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
	const int num_of_good_matches = matches.size()*good_match_percent;	// use instead some scoring mechenism, like matche score above threshold are selected
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

// returns alignment of two matrices
alignment_type get_alignment(Mat h1, Mat h2)
{
	// above the horizontal line
	int y1 = height/8;
	// below the horizontal line
	int y2 = (-1)*(height/8);
	// for x1, y1 getting corresponding point
	int y1_h1 = (h1.at<double>(1, 1)*y1 + h1.at<double>(1, 2))/(h1.at<double>(2, 1)*y1 + h1.at<double>(2, 2));
	int y1_h2 = (h2.at<double>(1, 1)*y1 + h2.at<double>(1, 2))/(h2.at<double>(2, 1)*y1 + h2.at<double>(2, 2));
	// for x2, y2 getting corresponding point
	int y2_h1 = (h1.at<double>(1, 1)*y2 + h1.at<double>(1, 2))/(h1.at<double>(2, 1)*y2 + h1.at<double>(2, 2));
	int y2_h2 = (h2.at<double>(1, 1)*y2 + h2.at<double>(1, 2))/(h2.at<double>(2, 1)*y2 + h2.at<double>(2, 2));

	// vertical check assuming y axis is facing up
	if(y1*y1_h1 < 0 and y1*y1_h2 > 0)	return VERTICAL_FIRST;		
	else if(y1*y1_h1 > 0 and y1*y1_h2 < 0)	return VERTICAL_SECOND;
	else if(y2*y2_h1 < 0 and y2*y2_h2 > 0)	return VERTICAL_SECOND;
	else if(y2*y2_h1 > 0 and y2*y2_h2 < 0)	return VERTICAL_FIRST;
	// horizontal check
//	else if(y1*y1_h1 > 0 or y1*y1_h2 > 0)	return HORIZONTAL;
	else					return NONE;
}
pair<int,vector<int> > getElementInRow(int** arr, int ind, vector<int> path, int max_ind){
	int j = 0;
	for(j=0; j<=max_ind && arr[ind][j] != 1; j++);
	if(j == max_ind + 1){
		return make_pair(ind,path);
	}
	else{
		path.push_back(j);
		pair<int, vector<int> >  ans = getElementInRow(arr,j,path, max_ind);
		return ans;
	}
}

bool comp(const pair<int,vector<int> > &p1, const pair<int,vector<int> >  &p2){
	if(p1.first == p2.first){
		vector<int> temp1 = p1.second;
		vector<int> temp2 = p2.second;
		return (temp1.size() > temp2.size());
	}
	else
		return (p1.first < p2.first);
}
vector<vector<int> > compute_rows(vector<pair<int,int> > horiz, int max_img_ind){
	int** adj_matrix;
	adj_matrix = new int*[max_img_ind+1];
	for(int i =0; i<max_img_ind+1;i++){
		int* temp = new int[max_img_ind+1];
		adj_matrix[i] = temp;
	}
	for(int i=0; i<horiz.size();i++){
		int start = horiz[i].first;
		int end = horiz[i].second;
		adj_matrix[start][end] = 1;
	}
	vector<pair<int, vector<int> > > dfs_answers;
	for(int i=0; i<=max_img_ind;i++){
		vector<int> temp;
		temp.push_back(i);
		pair<int,vector<int> > answer = getElementInRow(adj_matrix,i,temp, max_img_ind);
		dfs_answers.push_back(answer);
	}
	sort(dfs_answers.begin(), dfs_answers.end(),comp);
	int last = -1;
	vector<vector<int> > ret;
	for(int i=0; i<dfs_answers.size();i++){
		if((dfs_answers[i]).first != last){
			last = (dfs_answers[i]).first;
			ret.push_back(dfs_answers[i].second);
		}
	}
	cout << ret.size() << " NUmber of rows"<< endl;
	return ret;
} 

vector<vector<int> > getOrdering(vector<pair<int,int> > horiz, vector<pair<int,int> > vert, int img_max_ind){
	vector<vector<int> > hori_rows = compute_rows(horiz, img_max_ind);
	cout << hori_rows[0].size() << " waka" << hori_rows[1].size() << endl;
	int img_row_ind[img_max_ind+1];
	for(int i =0; i<hori_rows.size();i++){
		for(int j=0; j<hori_rows[i].size();j++)
			img_row_ind[hori_rows[i][j]] = i;
	}
	if(vert.size() == 0){
		return hori_rows;
	}
	else{
	vector<pair<int,int> > ordered_rows;
	int forward_edge[img_max_ind+1] = {-1};
	int backward_edge[img_max_ind+1] = {-1};
	for(int i=0; i<vert.size();i++){
		int start = vert[i].first;
		int end = vert[i].second;
		forward_edge[start] = end;
		backward_edge[end] = start;
		ordered_rows.push_back(make_pair(img_row_ind[start],img_row_ind[end]));
	}
	vector<vector<int> > vert_order = compute_rows(ordered_rows, hori_rows.size()-1);
//	cout << vert_order[0][0] << " Order of rows" << vert_order[0][1] << endl;
	int max_len = -1;
	int max_len_ind = 0;
	for(int i=0; i<hori_rows.size();i++){
//		cout << hori_rows[vert_order[0][i]].size() << " spaxe " << max_len << endl;
		int ind = vert_order[0][i];
		int val =  hori_rows[ind].size();
		if(max_len < val){
			max_len_ind = i;
			max_len = val;
			//cout << i << " Index of update" << endl;
		}	
	} 
	cout << max_len_ind << " Maximum length of row" << endl;
	int** final_answer;
	final_answer  = new int*[hori_rows.size()];
	for(int i =0; i<hori_rows.size();i++){
		int* temp = new int[max_len];
		final_answer[i] = temp;
	}
	for(int i=0; i<hori_rows.size();i++){
		for(int j=0; j<max_len;j++)
			final_answer[i][j] = -1;
	}
	for(int i=0; i<max_len;i++){
		int temp = vert_order[0][max_len_ind];
		final_answer[max_len_ind][i] = hori_rows[temp][i];
	}
	cout << final_answer[1][0] <<  " wow" <<endl;
	bool visited = false;
	for(int i = max_len_ind; i >=1; i--){
		int first_update_index = 0;
		for(int j=0; j<max_len;j++){
			if(final_answer[i][j] != -1){
				int temp = final_answer[i][j];
				cout << temp << " check" << endl;
				final_answer[i-1][j] = backward_edge[temp];
				cout << "police" << endl;
				if(!visited){
					first_update_index = j;
					visited = true;
				}
			}
		}
		int temp = vert_order[0][i-1];
		if(final_answer[i-1][first_update_index] != hori_rows[temp][0]){
			int ind = find(hori_rows[temp].begin(), hori_rows[temp].end(),final_answer[i-1][first_update_index]) - hori_rows[temp].begin();
			int temp1 = ind-1;
		       	for(int j = first_update_index-1; temp1 >=0 ;j--){
				final_answer[i-1][j] = hori_rows[temp][temp1];
				temp1--;
			}
		}	
	}
	for(int i = max_len_ind; i < vert_order[0].size()-1; i++){
		int first_update_index = 0;
		for(int j=0; j<max_len;j++){
			if(final_answer[i][j] != -1){
				int temp = final_answer[i][j];
				final_answer[i+1][j] = forward_edge[temp];
				first_update_index = j;
				break;
			}
		}
		int temp = vert_order[0][i+1];
		if(final_answer[i+1][first_update_index] != hori_rows[temp][0]){
			int ind = find(hori_rows[temp].begin(), hori_rows[temp].end(),final_answer[i+1][first_update_index]) - hori_rows[temp].begin();
			int temp1 = ind-1;
		       	for(int j = first_update_index-1; temp1 >=0 ;j--){
				final_answer[i+1][j] = hori_rows[temp][temp1];
				temp1--;
			}
		}	
	}
	vector<vector<int> > row_vec;
	for(int i =0; i<hori_rows.size(); i++){
		vector<int> temp;
		for(int j=0; j<max_len; j++)
			temp.push_back(final_answer[i][j]);
		row_vec.push_back(temp);
	}
	return row_vec;
}
}

// returns homography matrix between source and destination images
Mat get_homography(int r1, int c1, int r2, int c2, vector<vector<Mat>> &pair_wise_homography_matrix, vector<vector<int>> &ordered_rows)
{
	// base case
	if(r1==r2 and c1==c2)	return Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1;
	// destination is up from source
	if(r2 < r1)
	{}
	// destination is below from source
	else if(r2 > r1)
	{}
	// destination is in the same row
	else if(r2 == r1)
	{
		// destination is in the right side
		if(c2 > c1)
		{
			return 	get_homography(r1, c1+1, r2, c2, pair_wise_homography_matrix, ordered_rows) * 
				pair_wise_homography_matrix[ordered_rows[r1][c1]][ordered_rows[r1][c1+1]];
		}
		// destination is in the left side
		else
		{
			return 	get_homography(r1, c1-1, r2, c2, pair_wise_homography_matrix, ordered_rows) * 
				(pair_wise_homography_matrix[ordered_rows[r1][c1-1]][ordered_rows[r1][c1]].inv());
		}
	}
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
	//good_homography_indices.erase( good_homography_indices.begin()+image_count-1, good_homography_indices.end());
	
	// erasing elements having zero score.
//	auto i = good_homography_indices.begin();
//	for(; i<good_homography_indices.end(); i++)	if(i->first == 0)	break;
//	good_homography_indices.erase(i, good_homography_indices.end());

	// removing elements that does not make duplex
	int good_homography_indices_size = good_homography_indices.size();
	vector< pair< double, pair<int, int> >> duplex_homography_indices;
	for(int i=0; i<good_homography_indices_size-1; i++)
	{
		int x = good_homography_indices[i].second.first, y = good_homography_indices[i].second.second;
		for(int j=i+1; j<good_homography_indices_size; j++)
		{
			if(x == good_homography_indices[j].second.second and y == good_homography_indices[j].second.first)
			{
				duplex_homography_indices.push_back(good_homography_indices[i]);
				duplex_homography_indices.push_back(good_homography_indices[j]);
				break;
			}
		}
	}

	// printing sorted indices of homography
	cout << "after filtering\n";
	for(int i=0; i<duplex_homography_indices.size(); i++)	cout << duplex_homography_indices[i].first << ": " <<
							duplex_homography_indices[i].second.first << " -> " << 
							duplex_homography_indices[i].second.second << endl;
	
	// creating horizontal and vertical pair of indices
	vector< pair<int, int> > horizontal_indices, vertical_indices;
	for(int i=0; i<duplex_homography_indices.size(); i=i+2)
	{
		if( duplex_homography_indices[i].first >= 0.43 and duplex_homography_indices[i+1].first < 0.43)		
		{
			Mat h1 = pair_wise_homography_matrix[duplex_homography_indices[i].second.first][duplex_homography_indices[i].second.second];
			// above the horizontal line
			int y1 = 9*height/16;
			// below the horizontal line
			int y2 = (-1)*(9*height/16);
			// for x1, y1 getting corresponding point
			int y1_h1 = (h1.at<double>(1, 1)*y1 + h1.at<double>(1, 2))/(h1.at<double>(2, 1)*y1 + h1.at<double>(2, 2));
			// for x2, y2 getting corresponding point
			int y2_h1 = (h1.at<double>(1, 1)*y2 + h1.at<double>(1, 2))/(h1.at<double>(2, 1)*y2 + h1.at<double>(2, 2));
			cout << duplex_homography_indices[i].second.first << " -> " << duplex_homography_indices[i].second.second << endl;
			cout << y1 << " -> " << y1_h1 << " , " << y2 << " -> " << y2_h1 << endl;
			// vertical check assuming y axis is facing up
			if(y1*y1_h1 > 0 and y2*y2_h1 > 0 and y1_h1 < 2*y1)
				horizontal_indices.push_back(duplex_homography_indices[i].second); 
			continue;
		}
		else if(duplex_homography_indices[i+1].first >= 0.43 and duplex_homography_indices[i].first < 0.43)		
		{
			Mat h1 = pair_wise_homography_matrix[duplex_homography_indices[i+1].second.first][duplex_homography_indices[i+1].second.second];
			// above the horizontal line
			int y1 = 9*height/16;
			// below the horizontal line
			int y2 = (-1)*(9*height/16);
			// for x1, y1 getting corresponding point
			int y1_h1 = (h1.at<double>(1, 1)*y1 + h1.at<double>(1, 2))/(h1.at<double>(2, 1)*y1 + h1.at<double>(2, 2));
			// for x2, y2 getting corresponding point
			int y2_h1 = (h1.at<double>(1, 1)*y2 + h1.at<double>(1, 2))/(h1.at<double>(2, 1)*y2 + h1.at<double>(2, 2));
			cout << duplex_homography_indices[i+1].second.first << " -> " << duplex_homography_indices[i+1].second.second << endl;
			cout << y1 << " -> " << y1_h1 << " , " << y2 << " -> " << y2_h1 << endl;
			// vertical check assuming y axis is facing up
			if(y1*y1_h1 > 0 and y2*y2_h1 > 0 and y1_h1 < 2*y1)
				horizontal_indices.push_back(duplex_homography_indices[i+1].second); 
			continue;
		}

		alignment_type alignment = get_alignment(pair_wise_homography_matrix[duplex_homography_indices[i].second.first][duplex_homography_indices[i].second.second], 
							 pair_wise_homography_matrix[duplex_homography_indices[i+1].second.first][duplex_homography_indices[i+1].second.second]);

		if(alignment == VERTICAL_FIRST)
		{
			if(duplex_homography_indices[i].first >= 0.5 and duplex_homography_indices[i+1].first >= 0.5)	
				vertical_indices.push_back(duplex_homography_indices[i].second);
		}
		else if(alignment == VERTICAL_SECOND)
		{
			if(duplex_homography_indices[i].first >= 0.5 and duplex_homography_indices[i+1].first >= 0.5)	
				vertical_indices.push_back(duplex_homography_indices[i+1].second);
		}
	}
	cout << "*************\n";
	for(int i=0; i< horizontal_indices.size(); i++)		cout << horizontal_indices[i].first << " -> " << horizontal_indices[i].second << "  ";
	cout << endl;
	for(int i=0; i< vertical_indices.size(); i++)		cout << vertical_indices[i].first << " -> " << vertical_indices[i].second << "  ";
	cout << "-------------\n";
	
	// getting ordered rows
	vector<vector<int>> ordered_rows = getOrdering(horizontal_indices, vertical_indices, all_mappings.first.size()-1);

	cout << "----------\nordered_rows\n";
	// printing ordered rows
	int rows = ordered_rows.size();
	int cols = ordered_rows[0].size();
	cout << rows << " " << cols << endl;
	for(int i=0; i<rows; i++)
	{
		for(int j=0; j<cols; j++)
			cout << ordered_rows[i][j] << " ";
		cout << endl;
	}

	vector<Mat> colored_images = get_colored_images(folder_path);
	int vertical_row_threshold = 4*colored_images[0].rows;
	
	int j=0;
	// int pan_size = 5000;
	// Mat warp = Mat_<double>(5000,5000);
	// Mat trans = (Mat_<double>(3,3)<< 1,0,200,0,1,200,0,0,1);
	// Mat identity = (Mat_<double>(3,3)<< 1,0,0,0,1,0,0,0,1);
	// int mid = ordered_rows[j].size()/2;
	Mat warped_image = colored_images[ordered_rows[j][ordered_rows[j].size()-1]];
	// warpPerspective(warped_image,warp,trans*identity,warp.size());
	// imwrite("check12.jpg",warp);
	// Mat right_homo = identity; 
	// Size img_size = colored_images[ordered_rows[j][mid]].size();
	// for(int i = mid+1; i<ordered_rows[j].size();i++){
	// 	right_homo = right_homo * pair_wise_homography_matrix[ordered_rows[j][i-1]][ordered_rows[j][i]].inv();
	// 	warpPerspective(colored_images[ordered_rows[j][i]], warp, trans * right_homo, img_size);
	// 	imwrite("check"+ to_string(i)+".jpg",warp);
	// 	img_size.width *= 2;
	// }
	// Mat left_homo = identity;
	// for(int i = mid-1; i>=0;i--){
	// 	left_homo = left_homo * pair_wise_homography_matrix[ordered_rows[j][i]][ordered_rows[j][i+1]];
	// 	warpPerspective(colored_images[ordered_rows[j][i]], warp, trans * left_homo, img_size);
	// 	imwrite("check"+ to_string(i)+".jpg",warp);
	// 	img_size.width *= 2;
	// }
	for(int i=ordered_rows[j].size()-1; i>=1; i--)
	{
		if(ordered_rows[j][i] == -1)	continue;

		// warping the images
		Mat H_right_to_left = pair_wise_homography_matrix[ordered_rows[j][i-1]][ordered_rows[j][i]].inv();
		// coping panaroma to temp variable
		Mat left_image = colored_images[ordered_rows[j][i-1]];
		Mat right_image;
		warped_image.copyTo(right_image);
		
		Size output_size = Size(left_image.cols + right_image.cols, vertical_row_threshold);
		// plotting right image
		warpPerspective(right_image, warped_image, H_right_to_left, output_size);	
		imwrite("hello" + to_string(i) + ".jpg", warped_image);
		// getting left half of panorama == roi
		Mat half(warped_image, Rect(0, 0, left_image.cols, left_image.rows));
		// coping left image to it
		left_image.copyTo(half);
	}	
	warped_image = equalize_hist(warped_image);
	imwrite("warped_image.jpg", warped_image);
	return 0;
}

		// warping the images
//		Mat H_right_to_left = pair_wise_homography_matrix[ordered_rows[0][0]][ordered_rows[0][1]].inv();
//		Mat left_image = colored_images[ordered_rows[0][0]];
//		Mat right_image = colored_images[ordered_rows[0][1]];
//		Mat warped_image;
//		Size output_size = Size(left_image.cols + right_image.cols, left_image.rows);
//		// plotting right image
//		warpPerspective(right_image, warped_image, H_right_to_left, output_size);
//		// getting left half of panorama == roi
//		Mat half(warped_image, Rect(0, 0, left_image.cols, left_image.rows));
//		// coping left image to it
//		left_image.copyTo(half);
