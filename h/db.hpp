#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <unordered_map>
#include <numeric>
#include "glob_img.hpp" 

using namespace std;
using namespace cv;
using namespace cv::ml;

class DB {

public:
	Mat labels, data;

	// constructor
	DB(
	   String db_path
	   );

	// generate dictionary from images,
	// given the k-means K parameter
	int make(vector<String> data_path, 
			 Ptr<Feature2D> detector, 
	   	 	 Ptr<Feature2D> descriptor,
			 int K
			);

	// open existing dictionary
	int open();

	// get histogram (feature vector)
	int get_hist(Mat img, Mat& hist);
	int get_hist(vector<String> img_paths, Mat& hists);

	// SVM classification
	int train_svm(vector<String> dir_paths);
	int predict_svm(String dir_path, Mat& results);

private:
	String db_path;
	Ptr<SVM> svm;
	Ptr<Feature2D> detector, descriptor;
	int K;
	int write(); // write data to yml dictionary
	int myHist(const Mat*,Mat&);
	
	unordered_map< string,Ptr<Feature2D> > features_map = {{"ORB", ORB::create()},
		{"BRISK", BRISK::create()},
		{"AKAZE", AKAZE::create()}
	};
};


