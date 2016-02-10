#include "db.hpp"

using namespace std;
using namespace cv;

DB::DB(String db_path_)
:  db_path(db_path_)
{}

int DB::make(vector<String> img_paths,
		Ptr<Feature2D> detector_,
		Ptr<Feature2D> descriptor_,
		int K_)
{
	detector = detector_;
	descriptor = descriptor_;
	K = K_;

	//-------------------
	// Init
	//-------------------

	vector<KeyPoint> keypoints;
	vector<Mat> descriptors_list;
	Mat descriptors;

	//-------------------
	// Gather descriptors
	//-------------------

	for(int i=0; i<img_paths.size(); i++)
	{
		// Load
		Mat img_temp = imread( img_paths[i], CV_LOAD_IMAGE_GRAYSCALE );
		if (img_temp.empty())
		{
			cout << "discarded " << img_paths[i] << endl;
			continue;
		}

		// Detect
		vector<KeyPoint> keypoints_temp;
		detector->detect(img_temp, keypoints_temp, Mat());
		keypoints.insert(keypoints.end(),
				keypoints_temp.begin(),
				keypoints_temp.end());

		// Describe
		Mat descriptors_temp;
		descriptor->compute( img_temp, keypoints_temp, descriptors_temp);
		descriptors_list.push_back(descriptors_temp);
	}

	vconcat(descriptors_list, descriptors);

	//-------------------
	// Cluster
	//-------------------

	Mat descriptors_float,
	    labels_,
	    centers;
	descriptors.convertTo(descriptors_float,CV_32FC1);
	TermCriteria criteria(TermCriteria::EPS+TermCriteria::COUNT, 1.0, 1.0);
	int K = 5,
	    attempts = 1,
	    flags = KMEANS_PP_CENTERS;

	kmeans(descriptors_float, K, labels_, criteria, attempts, flags, centers);

	data = descriptors;
	labels = labels_;
	
	//-------------------
	// Write to database
	//-------------------

	DB::write();

	return 0;

}


int DB::get_hist(Mat img, Mat& hist)
{

	// Detect & Describe
	vector<KeyPoint> keypoints;
	Mat descriptors;
	detector->detect(img, keypoints, Mat());
	descriptor->compute( img, keypoints, descriptors);

	// Find closest neighbours' indexes
	// & Make histogram of indexes

	// Get matches
	BFMatcher matcher(NORM_HAMMING);
	vector< DMatch > matches;
	matcher.match( descriptors, data, matches );
	
	Mat labels_;
	transpose(labels,labels_);
	hist = Mat::zeros(K+2,1,CV_32SC(1));
	int* p_l = labels_.ptr<int>(0); // labels pointer
	int* p_h = hist.ptr<int>(0);	// hist pointer
	
	for(int i=0; i<matches.size(); i++)
	{
		int idx = matches[i].trainIdx; // get index of label 
		if(idx < labels.rows)
		{
			p_h[p_l[idx]] ++;
		}
		else
		{
			cerr << idx << endl;
		}
	}

	return 0;
}

int DB::get_hist(vector<String> img_paths, Mat& hists)
{
	Mat hist;
	vector<Mat> hist_vect;
	for(int i=0; i<img_paths.size(); i++)
	{
		// Load img
		Mat img = imread(img_paths[i], CV_LOAD_IMAGE_GRAYSCALE);
		if (img.empty())
		{
			cout << "discarded " << img_paths[i] << endl;
			continue;
		}

		// get histogram and add
		get_hist(img, hist);
		hist_vect.push_back(hist.clone());
	}

	hconcat(hist_vect,hists);
	normalize(hists,hists);

	return 0;
}


