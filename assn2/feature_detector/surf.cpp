#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void store_image(Mat img, string img_name)
{
	imwrite(img_name, img);
}

vector<KeyPoint> surf(Mat src_img, int hessian_threshold)
{
	Ptr<SURF> detector = SURF::create( hessian_threshold );
	vector<KeyPoint> keypoints;
	detector->detect(src_img, keypoints);
	// vector of key points
	return keypoints;
}

// this function reads all images from folder path given in the arg
// and returns 2D vector whose (i,j) th element represents
// jth feature of ith image
vector<vector<KeyPoint>> get_key_points(string folder_path)
{
	vector<vector<KeyPoint>> image_key_points;

	// complete path of the images
	vector<cv::String> img_file_names;
	// reading the file names
	glob(folder_path + "/*.jpg", img_file_names, false);
	int images_size = img_file_names.size();
	// traversing the images
	for(int i=0; i<images_size; i++)
	{
		Mat src_image = imread(img_file_names[i], IMREAD_GRAYSCALE);
		if(src_image.empty())	cout << "Can not read image...\n";
		// hessian threshold: 3000
		vector<KeyPoint> src_key_points = surf(src_image, 3000);
		image_key_points.push_back( src_key_points );
		// writing the image
		Mat marked_image;
		drawKeypoints(src_image, src_key_points, marked_image);
		store_image(marked_image, to_string(i)+".jpg");
	}

	return image_key_points;
}

int main(int argc, char* argv[])
{
	// getting the image folder path
	string folder_path;
	cout << "Enter path to folder:\n" << 
		"default path is: /home/dwijesh/Documents/sem5/vision/assns/assn2_data/1" 
		<< endl;
	cin >> folder_path;
	// for ith image, vector of its key points.
	vector<vector<KeyPoint>> all_key_points = get_key_points(folder_path);
	return 0;
}
