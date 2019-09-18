#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

	using namespace cv;
	using namespace cv::xfeatures2d;
	using namespace std;

	int main(int argc, char* argv[])
	{
		// reading image file
		string img_path;
		if(argc == 1)	{ cout << "Expecting command line input: img path\n"; return 0;}
		img_path = argv[1];
		Mat src = imread(img_path, IMREAD_GRAYSCALE);
		if(src.empty())		{ cout << "File not found.\n"; return 0;}

		// detecting keypoits using surf detector
		int minHessian = 200;
		Ptr<SURF> detector = SURF::create( minHessian );
		vector<KeyPoint> keypoints;
		detector->detect(src, keypoints);

		// drawing keypoints
		Mat img_keypoints;
		drawKeypoints(src, keypoints, img_keypoints);

		// show detected keypoints
		imshow("SURF keypoints", img_keypoints);

		waitKey(0);
		return 0;
	}
#else
	int main()
	{
		std::cout << "xfeatures contrib module is needed.\n";
		return 0;
	}
#endif
