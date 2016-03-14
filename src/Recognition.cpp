#include<opencv2/opencv.hpp>
#include"Train.h"

using namespace std;
using namespace cv;

void otsuMethod (string path) {
	Mat dstOtsu, drawing, img, img2;
	vector<vector<Point> > contours;
	vector<vector<Point> > contours2;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	double min, max;
	string path2;
	path2 = "images/" + path;
	img = imread(path2, CV_LOAD_IMAGE_GRAYSCALE);

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
			contours2.push_back(contours[i]); //Se añade a contours2. Aqui solo se almacenan los blobs válidos por tamaño
		}
	}

	imshow( "Blobs", drawing );

	//Metodo adaptativo (demostrado peor para imagenes dadas en la practica)
	//adaptiveThreshold(img, dstAdaptive, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 69, 1);
	//adaptiveThreshold(img2, dstAdaptive2, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 69, 1);
	//imshow("Adaptive",dstAdaptive);
	//imshow("Adaptive 2", dstAdaptive2);

	waitKey(0);

	//Se calculan los momentos de la imagen
	vector<Moments> mu(contours.size() );
	vector<double> ourDescriptors;
	double hu[7];
	for( int i = 0; i < contours2.size(); i++ ){
		cout << "Contorno " << i << ":\n";
		mu[i] = moments( contours2[i], false );
		HuMoments(mu[i], hu);
		cout << "\tArea: " << mu[i].m00 <<
				"\n\tPerimetro: " << arcLength(contours2[i], true) <<
				"\n\tMomento Inv. 1: " << hu[0] <<
				"\n\tMomento Inv. 2: " << hu[1] <<
				"\n\tMomento Inv. 3: " << hu[2] << "\n";
		ourDescriptors.push_back(mu[i].m00);
		ourDescriptors.push_back(arcLength(contours2[i], true));
		ourDescriptors.push_back(hu[0]);
		ourDescriptors.push_back(hu[1]);
		ourDescriptors.push_back(hu[2]);
	}

	//Se calcula la distancia de Mahalanobis
	FileStorage fs2("training.yml", FileStorage::READ);
	vector<double> muMeans(5);
	vector<double> muVariances(5);

	fs2["ruedaMedia"] >> muMeans;
	fs2["ruedaVarianza"] >> muVariances;
	fs2.release();

	double mahalanobis = 0;
	for( int i = 0; i < ourDescriptors.size(); i++ ){
		cout << ourDescriptors[i] << " " << muMeans[i] << " " << muVariances[i] << "\n";
		mahalanobis += pow(ourDescriptors[i] - muMeans[i],2)/pow(muVariances[i],2);
	}
	cout << "Distancia de Mahalanobis: " << mahalanobis;
}


int main (int argc, const char* argv[]){
	for(int i=0; i<argc;i++){
		cout << argv[i] << endl;
	}
	if(strcmp(argv[1], "train") == 0){
		remove("training.yml");
		train("images/circulo", 5, "circulo", "write");
		train("images/rectangulo", 5, "rectangulo", "append");
		train("images/triangulo", 5, "triangulo", "append");
		train("images/vagon", 5, "vagon", "append");
		train("images/rueda", 5, "rueda", "append");
	}else if(strcmp(argv[1], "recognition") == 0){
		otsuMethod(argv[2]);
	}

}
