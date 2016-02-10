#include "db.hpp"

using namespace std;
using namespace cv;

int DB::write()
{

	FileStorage db(db_path, FileStorage::WRITE);

	// write labels
	db << "labels";
	db << labels;

	// write descriptors
	db << "data";
	db << data;

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

	// get labels
	node = db["labels"];
	node >> labels;

	// get descriptors
	node = db["data"];
	node >> data;

	return 0;
}
