#include<opencv2/opencv.hpp>
#include"Train.h"

using namespace std;
using namespace cv;

int main (int argc, const char* argv[]){
	for(int i=0; i<argc;i++){
		cout << argv[i] << endl;
	}
	if(strcmp(argv[1], "train") == 0){
		cout << "IEEEEEE";
		train("images/circulo", 5, "circulo");
		train("images/rectangulo", 5, "rectangulo");
		train("images/triangulo", 5, "triangulo");
		train("images/vagon", 5, "vagon");
	}
}

void otsuMethod (string path) {
	Mat dstOtsu, drawing, img, img2;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	double min, max;
	img = imread("images/rueda1.pgm", CV_LOAD_IMAGE_GRAYSCALE);

	//Otsu's thresholding
	threshold( img, dstOtsu, 0, 255, THRESH_OTSU+THRESH_BINARY_INV );
	imshow("Otsu",dstOtsu);
	findContours(dstOtsu, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
	drawing = Mat(dstOtsu.size(), CV_8UC3);
	for( int i = 0; i< contours.size(); i++ ){
		if ( cv::contourArea(contours[i], false) > 40 && contourArea(contours[i], false) < drawing.rows*drawing.cols){
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
			for(int col=0; col<drawing.cols; col++){
				for(int row=0; row<drawing.rows; row++){
					if (pointPolygonTest(contours[i],Point(col,row),false) == 1){
						floodFill(drawing, Point(col,row), color, (cv::Rect*)0, cv::Scalar(), 200);
					}
				}
			}

		}
	}

	imshow( "Blobs", drawing );

	//Metodo adaptativo (demostrado peor para imagenes dadas en la practica)
	/*adaptiveThreshold(img, dstAdaptive, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 69, 1);
	adaptiveThreshold(img2, dstAdaptive2, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 69, 1);
	imshow("Adaptive",dstAdaptive);
	imshow("Adaptive 2", dstAdaptive2);*/

	//Se calculan los momentos de la imagen
	vector<Moments> mu(contours.size() );
	for( int i = 0; i < contours.size(); i++ ){
		cout << "Contorno " << i << ":\n";
		mu[i] = moments( contours[i], false );
		cout << "\tArea: " << mu[i].m00 << "\n\tPerimetro: " << arcLength(contours[i], true) <<
				"\n\tMomento Inv.1: " << mu[i].m20+mu[i].m02 <<
				"\n\tMomento Inv.2: " << pow(mu[i].m20-mu[i].m02,2) + 4*pow(mu[i].m11,2) <<
				"\n\tMomento Inv.3: " << pow(mu[i].m30-mu[i].m12,2) + pow(mu[i].m21-mu[i].m03,2) << "\n";
	}

	waitKey(0);

}
