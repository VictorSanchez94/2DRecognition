#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void train(string path, int numObjects, string objectType, char* mode) {
	Mat dstOtsu, drawing, img, img2;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	double min, max;
	vector<double> muMeans(5);
	vector<double> muVariances(5);
	double hu[7];

	for (int i=1; i<=numObjects; i++){
		stringstream ss;
		ss << path << i << ".pgm";
		img = imread(ss.str(), CV_LOAD_IMAGE_GRAYSCALE);

		//Otsu's thresholding
		threshold( img, dstOtsu, 0, 255, THRESH_OTSU+THRESH_BINARY_INV );
		findContours(dstOtsu, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
		drawing = Mat(dstOtsu.size(), CV_8UC3);

		//Se calculan los momentos de la imagen
		vector<Moments> mu(contours.size() );

		for( int i = 0; i < contours.size(); i++ ){
			//cout << "Contorno " << i << ":\n";
			mu[i] = moments( contours[i], false );
			HuMoments(mu[i], hu);
			muMeans[0] += mu[i].m00;
			muMeans[1] += arcLength(contours[i], true);
			muMeans[2] += hu[0];
			muMeans[3] += hu[1];
			muMeans[4] += hu[2];
		}
	}
	//Se calcula la media
	for (int i=0; i<muMeans.size(); i++){
		muMeans[i] = muMeans[i]/numObjects;
	}


	//Se calcula la varianza
	for (int i=1; i<=numObjects; i++){
		stringstream ss;
		ss << path << i << ".pgm";
		img = imread(ss.str(), CV_LOAD_IMAGE_GRAYSCALE);

		//Otsu's thresholding
		threshold( img, dstOtsu, 0, 255, THRESH_OTSU+THRESH_BINARY_INV );
		findContours(dstOtsu, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
		drawing = Mat(dstOtsu.size(), CV_8UC3);

		//Se calculan los momentos de la imagen
		vector<Moments> mu(contours.size() );
		for( int i = 0; i < contours.size(); i++ ){
			//cout << "Contorno " << i << ":\n";
			mu[i] = moments( contours[i], false );
			HuMoments(mu[i], hu);
			muVariances[0] += pow(mu[i].m00 - muMeans[0],2);
			muVariances[1] += pow(arcLength(contours[i], true) - muMeans[1],2);
			muVariances[2] += pow(hu[0] - muMeans[2],2);
			muVariances[3] += pow(hu[1] - muMeans[3],2);
			muVariances[4] += pow(hu[2] - muMeans[4],2);
		}
	}

	//Se calcula la varianza
	for (int i=0; i<muMeans.size(); i++){
			muVariances[i] = muVariances[i]/numObjects;
	}

	//TODO: Muy cutre esto
	if(strcmp(mode, "write") == 0){
		FileStorage fs("training.yml", FileStorage::WRITE);
		stringstream media;
		media << objectType << "Media";
		stringstream varianza;
		varianza << objectType << "Varianza";
		fs << media.str()  << muMeans;
		fs << varianza.str() << muVariances;
		fs.release();
	}else{
		FileStorage fs("training.yml", FileStorage::APPEND);
		stringstream media;
		media << objectType << "Media";
		stringstream varianza;
		varianza << objectType << "Varianza";
		fs << media.str()  << muMeans;
		fs << varianza.str() << muVariances;
		fs.release();
	}
}
