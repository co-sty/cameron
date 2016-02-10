#include "db.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

int DB::train_svm(vector<String> img_dirs)
{
	// format training data
	vector<String> img_paths;
	vector<int> markers_v;
	for(int i=0; i<img_dirs.size(); i++)
	{
		// paths to images
		vector<String> imgs;
		glob_img(img_dirs[i],imgs);
		img_paths.insert(img_paths.end(),
				imgs.begin(),
				imgs.end());


		// references
		cout << "[ " << i << " for " << img_dirs[i] << " ]" << endl;
		vector<int> m(imgs.size());
		fill(m.begin(),m.end(),i);
		markers_v.insert(markers_v.end(),m.begin(),m.end());
	}

	Mat training_hists;
	get_hist(img_paths,training_hists);
	transpose(training_hists,training_hists);
	training_hists.convertTo(training_hists,CV_32F);
	Mat markers(markers_v,false);
	Ptr<TrainData> training_data = TrainData::create(training_hists,ROW_SAMPLE,markers);

	// Create SVM
	svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::POLY); // or SVM::LINEAR
	svm->setDegree(0.5);
	svm->setGamma(1);
	svm->setCoef0(1);
	svm->setNu(0.5);
	svm->setP(0);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 0.01));
	svm->setC(1); // 1 or 2

	// Train SVM
	svm->train(training_data);

	return 0;
}

int DB::predict_svm(String dir_path, Mat& response)
{
	vector<String> img_paths;
	glob_img(dir_path,img_paths);
	Mat hists;
	get_hist(img_paths,hists);
	transpose(hists,hists);
	hists.convertTo(hists,CV_32F);

	svm->predict(hists,response);
	transpose(response,response);

	return 0;
}
