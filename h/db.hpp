#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <unordered_map>

using namespace std;
using namespace cv;

class DB {

public:
	vector<Mat> data;

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
	int get_hist(String img_path);
	int get_hist(vector<String> img_paths);

private:
	String db_path;
	Ptr<Feature2D> detector, descriptor;
	int K;
	int write(); // write data to yml dictionary
	unordered_map< string,Ptr<Feature2D> > features_map = {{"ORB", ORB::create()},
		{"BRISK", BRISK::create()},
		{"AKAZE", AKAZE::create()}
	};
};


