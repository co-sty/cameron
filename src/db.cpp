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
		Mat img_temp = imread( img_paths.at(i), CV_LOAD_IMAGE_GRAYSCALE );

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

	// k-means with Hamming distance
	// -> data

	Mat a = Mat_<double>::eye(3, 3);
	vector<Mat> b;
	b.push_back(a);
	b.push_back(a);
	b.push_back(a);
	data = b;

	DB::write();

	return 0;

}

int DB::open()
{
    vector<Mat> data_;

	FileStorage db(db_path, FileStorage::READ);

	if (!db.isOpened())
    {
      cerr << "failed to open " << db_path << endl;
      return 1;
    }

    FileNode node;
/*
    // get detector/descriptor names
    node = db["meta"];
    FileNodeIterator it = node.begin();
    detector = features_map[(string)*it];
    it++;
    */

    // get actual data
    node = db["data"];
    if (node.type() != FileNode::SEQ)
    {
      cerr << "data is not a sequence" << endl;
      return 1;
    }

    node >> data_;

    for(int i=0; i<data_.size(); i++)
    	cout<<data_[i]<<endl;

    data = data_;

    return 0;
}

int DB::write()
{

	FileStorage db(db_path, FileStorage::WRITE);
	/*
    db << "meta" << "[";
    db << features_map[detector];
    db << features_map[descriptor];
    db << "]";
	*/
    // write bag of features

    db << "data" << "[";

    for(int i=0; i<DB::data.size(); i++)
    	db << DB::data.at(i);

    db << "]";

    return 0;
}

int DB::get_hist(vector<String> img_paths)
{
	for(int i=0; i<img_paths.size(); i++)
		int j;
	return 0;
}

int DB::get_hist(String img_path)
{
	Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
	return 0;
}
