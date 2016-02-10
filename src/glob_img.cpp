#include "glob_img.hpp"

using namespace std;
using namespace cv;


int glob_img(String dir, vector<String>& img_paths)
{
	// get files in dir	
	glob(dir,img_paths);

	// remove non-images
	remove_if(img_paths.begin(),img_paths.end(),[&](String s){
			bool x = (s.find(".jpg")==string::npos);
			x = x && (s.find(".png")==string::npos);
			if(x)
			cout << "rm " << s << endl;
			return x;
			});
	return 0;
}

int glob_img(vector<String> dir, vector<String>& img_paths)
{
	// loop over vector
	vector<String> img_paths_temp;
	for(int i=0; i<dir.size(); i++)
	{
		glob_img(dir[i], img_paths_temp);
		img_paths.insert(img_paths.end(),
				img_paths_temp.begin(),
				img_paths_temp.end());
	}

	return 0;
}
