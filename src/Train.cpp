#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void train(string path, int numObjects, string objectType) {
	Mat dstOtsu, drawing, img, img2;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	double min, max;
	vector<double> muMeans(5);

	for (int i=1; i<=numObjects; i++){
		stringstream ss;
		ss << path << i << ".pgm";
		img = imread(ss.str(), CV_LOAD_IMAGE_GRAYSCALE);

		//Otsu's thresholding
		threshold( img, dstOtsu, 0, 255, THRESH_OTSU+THRESH_BINARY_INV );
		imshow("Otsu",dstOtsu);
		findContours(dstOtsu, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
		drawing = Mat(dstOtsu.size(), CV_8UC3);
		/*for( int i = 0; i< contours.size(); i++ ){
			if ( cv::contourArea(contours[i], false) > 40 && contourArea(contours[i], false) < drawing.rows*drawing.cols){
				Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
			}
		}*/

		//Se calculan los momentos de la imagen
		vector<Moments> mu(contours.size() );
		for( int i = 0; i < contours.size(); i++ ){
			//cout << "Contorno " << i << ":\n";
			mu[i] = moments( contours[i], false );
			/*cout << "\tArea: " << mu[i].m00 << "\n\tPerimetro: " << arcLength(contours[i], true) <<
					"\n\tMomento Inv.1: " << mu[i].m20+mu[i].m02 <<
					"\n\tMomento Inv.2: " << pow(mu[i].m20-mu[i].m02,2) + 4*pow(mu[i].m11,2) <<
					"\n\tMomento Inv.3: " << pow(mu[i].m30-mu[i].m12,2) + pow(mu[i].m21-mu[i].m03,2) << endl;*/
			muMeans[0] += mu[i].m00;
			muMeans[1] += arcLength(contours[i], true);
			muMeans[2] += mu[i].m20+mu[i].m02;
			muMeans[3] += pow(mu[i].m20-mu[i].m02,2) + 4*pow(mu[i].m11,2);
			muMeans[4] += pow(mu[i].m30-mu[i].m12,2) + pow(mu[i].m21-mu[i].m03,2);
		}

	}
	for (int i=0; i<muMeans.size(); i++){
		muMeans[i] = muMeans[i]/numObjects;
	}
	FileStorage fs("training.yml", FileStorage::APPEND);
	fs << objectType << muMeans;
	fs.release();

}
