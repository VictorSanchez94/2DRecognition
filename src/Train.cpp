#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void train(string path, int numObjects, string objectType, char* mode) {
	//Inicializacion de variables
	Mat dstOtsu, drawing, img, img2;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	double min, max;
	vector<double> muMeans(4);
	vector<double> muVariances(4);
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
			mu[i] = moments( contours[i], false );
			HuMoments(mu[i], hu);
			muMeans[0] += mu[i].m00;
			muMeans[1] += arcLength(contours[i], true);
			//muMeans[2] += hu[0];
			muMeans[2] += hu[1];
			muMeans[3] += hu[2];
		}
	}
	//Se calcula la media
	for (int i=0; i<muMeans.size(); i++){
		muMeans[i] = muMeans[i]/numObjects;
	}

	vector<double> muVariances1(5);
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
			if ( cv::contourArea(contours[i], false) > 60 && contourArea(contours[i], false) < drawing.rows*drawing.cols){
				mu[i] = moments( contours[i], false );
				HuMoments(mu[i], hu);
				muVariances[0] += pow(mu[i].m00 - muMeans[0],2);
				muVariances[1] += pow(arcLength(contours[i], true) - muMeans[1],2);
				//muVariances[2] += pow(hu[0] - muMeans[2],2);
				muVariances[2] += pow(hu[1] - muMeans[2],2);
				muVariances[3] += pow(hu[2] - muMeans[3],2);
			}
		}
		/*if(i==1){ //Se almacenan aparte los descriptores de la primera imagen para regularizar
			muVariances1[0] = muVariances[0];
			muVariances1[1] = muVariances[1];
			muVariances1[2] = muVariances[2];
			muVariances1[3] = muVariances[3];
			muVariances1[4] = muVariances[4];
			muVariances[0] = 0;
			muVariances[1] = 0;
			muVariances[2] = 0;
			muVariances[3] = 0;
			muVariances[4] = 0;
		}*/
	}

	//Se calcula la varianza regularizada
	for (int i=0; i<muMeans.size(); i++){
		//muVariances[i] = muVariances[i]/(numObjects-1);
		muVariances[i] = pow(0.01*muVariances[i],2)/numObjects + muVariances[i]*(numObjects-1)/numObjects;
	}

	//Guardar entrenamiento en fichero
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
