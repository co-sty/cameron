#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "db.hpp"

using namespace cv;
using namespace std;

void readme();

/** @function main */
int main( int argc, char** argv )
{
	// get input arguments
	vector<String> img_paths;	// images
	for(int i=1; i<argc-1; i++)
	{
		img_paths.push_back(argv[i]);
	}

	String db_name = argv[argc-1]; // database
	Mat check_mat = imread(db_name); 
  	if(!check_mat.empty()){ return -1;}

	// make detectors
	Ptr<Feature2D> detector = AKAZE::create();
	Ptr<Feature2D> descriptor = AKAZE::create();
	cout<<detector->descriptorType()<<endl;

	// make database
	DB db (db_name);
	db.make(img_paths, detector,descriptor ,3);

	// print data
	for(int i=0; i<db.data.size(); i++)
		cout << db.data.at(i) << endl;

	return 0;
}

/** @function readme */
void readme()
{ 
	cout << " Usage: ./cameron <img1> ... <imgN> <database_name>" << endl;
	cout << " e.g. : ./cameron data/train/+/1.jpg data/train/+/2.jpg data/db.yml" << endl; 
}
