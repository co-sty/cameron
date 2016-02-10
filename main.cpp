#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "db.hpp"
#include "glob_img.hpp"

using namespace cv;
using namespace std;


void filter_images(vector<String>& img_paths);
void readme();

/** @function main */
int main( int argc, char** argv )
{
	if(argc < 1)
	{
		readme();
		return -1;
	}

	// get input arguments
	vector<String> img_paths;	// images
	for(int i=1; i<argc-1; i++)
	{
		img_paths.push_back(argv[i]);
	}

	String db_name = argv[argc-1]; // database
	Mat check_mat = imread(db_name);
  	if(!check_mat.empty()){ return -1;}
	// Overriding input
	db_name = "data/db.yml";
	vector<String> all_imgs, 
	      train_dirs{(String)"data/train/+",(String)"data/train/-"};
	glob("data/train/",all_imgs,true);
	String test_dir_p = "data/test/+/",
	       test_dir_m = "data/test/-/";
	for (int i=0; i< train_dirs.size(); i++)
		cout << train_dirs[i] << endl;

	// make detectors
	Ptr<Feature2D> detector = BRISK::create();
	Ptr<Feature2D> descriptor = detector;//AKAZE::create();

	// make database
	cout << " ––– Making database at " << db_name << " ..." << endl;
	int K = 4;
	DB db (db_name);
	db.make(all_imgs, detector,descriptor , K);

	// SVM classification
	// training
	cout << " ––– Training SVM ..." <<endl;
	db.train_svm(train_dirs);
	// prediction
	cout << " ––– Predicting ..." <<endl;
	Mat res;
	cout << "\t+ prediction (" << test_dir_p << ") :"<< endl;
	db.predict_svm(test_dir_p,res);
	cout << "\t" << res << endl;

	cout << "\t- prediction (" << test_dir_m << ") :"<< endl;
	db.predict_svm(test_dir_m,res);
	cout << "\t" << res << endl;

	return 0;
}

/** @function readme */
void readme()
{
	cout << " Usage: ./cameron <img1> ... <imgN> <database_name>" << endl;
	cout << " e.g. : ./cameron data/train/+/1.jpg data/train/+/2.jpg data/db.yml" << endl;
}

