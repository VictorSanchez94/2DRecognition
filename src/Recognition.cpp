#include<opencv2/opencv.hpp>
#include"Train.h"

using namespace std;
using namespace cv;

int TIPOS_FIGURAS = 5;
string tipos[] = {"circulo", "rectangulo", "rueda", "triangulo", "vagon"};
double CHI_2_TEST = 9.49;	//0.0.5 error		//13.28;

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
	imshow("Imagen Original", img);

	//Otsu's thresholding
	threshold( img, dstOtsu, 0, 255, THRESH_OTSU+THRESH_BINARY_INV );
	imshow("Otsu",dstOtsu);
	findContours(dstOtsu, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
	drawing = Mat(dstOtsu.size(), CV_8UC3);
	for( int i = 0; i< contours.size(); i++ ){
		if ( cv::contourArea(contours[i], false) > 60 && contourArea(contours[i], false) < drawing.rows*drawing.cols){
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
			vector<Point> tmp = contours.at(i);
			const Point* elementPoints[1] = {&tmp[0]};
			int numberOfPoints = (int)tmp.size();
			fillPoly(drawing, elementPoints, &numberOfPoints, 1, color, 8);
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
	Mat contourRecognition = Mat(dstOtsu.size(), CV_8UC3);
	for( int i = 0; i < contours2.size(); i++ ){
		mu[i] = moments( contours2[i], false );
		HuMoments(mu[i], hu);
		ourDescriptors.push_back(mu[i].m00);
		ourDescriptors.push_back(arcLength(contours2[i], true));
		//ourDescriptors.push_back(hu[0]);
		ourDescriptors.push_back(hu[1]);
		ourDescriptors.push_back(hu[2]);
	}

	//Se calcula la distancia de Mahalanobis
	FileStorage fs2("training.yml", FileStorage::READ);
	vector<double> muMeans(4);
	vector<double> muVariances(4);

	for (int x=0; x<contours2.size(); x++){
		bool moreThanOne = false;
		double bestMahalanobis = HUGE_VAL;
		string bestObjectCandidate = "";
		drawContours( contourRecognition, contours2, x, Scalar(0,255,0), 1, 8, hierarchy, 0, Point() );
		vector<Point> tmp = contours2.at(x);
		const Point* elementPoints[1] = {&tmp[0]};
		int numberOfPoints = (int)tmp.size();
		fillPoly(drawing, elementPoints, &numberOfPoints, 1, Scalar(0,255,0), 8);
		cout << "\nContorno " << x+1 << ":" << endl;
		int maxDescriptor = x*4;
		for (int j=0; j<TIPOS_FIGURAS; j++){

			stringstream ssMedia, ssVarianza;
			ssMedia << tipos[j] << "Media";
			ssVarianza << tipos[j] << "Varianza";

			fs2[ssMedia.str()] >> muMeans;
			fs2[ssVarianza.str()] >> muVariances;

			double mahalanobis = 0;

			//Suma de las distancias de Mahalanobis entre cada descriptor con el de train
			for( int i = maxDescriptor; i < maxDescriptor+4; i++ ){
				mahalanobis += pow(ourDescriptors[i] - muMeans[i%4],2)/muVariances[i%4];
			}
			if(mahalanobis < bestMahalanobis){
				if(mahalanobis <= CHI_2_TEST && bestMahalanobis <= CHI_2_TEST){
					moreThanOne = true;
				}
				bestMahalanobis = mahalanobis;
				bestObjectCandidate = tipos[j];
			}
			cout << "Distancia de Mahalanobis con " << tipos[j] << ": " << mahalanobis << endl;

		}
		//Posibles casos de distincion
		if(bestMahalanobis <= CHI_2_TEST && !moreThanOne){
			cout << "== El contorno " << x+1 << " es un " << bestObjectCandidate << ". ==" << endl;
		}else if(bestMahalanobis <= CHI_2_TEST && moreThanOne){
			cout << "== El contorno " << x+1 << " se parece a más de un objeto conocido. ==" << endl;
		}else{
			cout << "== El contorno " << x+1 << " es un objeto desconocido. ==" << endl;
		}
		imshow("Contorno a reconocer", contourRecognition);
		waitKey(0);
	}
	fs2.release();
}


int main (int argc, const char* argv[]){
	if(strcmp(argv[1], "train") == 0){
		cout << "Entrenando..." << endl;
		train("images/circulo", 5, "circulo", "write");
		train("images/rectangulo", 5, "rectangulo", "append");
		train("images/triangulo", 5, "triangulo", "append");
		train("images/vagon", 5, "vagon", "append");
		train("images/rueda", 5, "rueda", "append");
		cout << "Fin del entrenamiento. Se ha generado el fichero 'training.yml'." << endl;
	}else if(strcmp(argv[1], "recognition") == 0){
		otsuMethod(argv[2]);
	}
}
