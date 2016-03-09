#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main (){

//void otsuMethod () {
	Mat dstOtsu, drawing,/*dstOtsu2, dstAdaptive, dstAdaptive2, */img, img2;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	double min, max;
	img = imread("images/rectangulo2.pgm", CV_LOAD_IMAGE_GRAYSCALE);
	//img2 = imread("images/vagon1.pgm", CV_LOAD_IMAGE_GRAYSCALE);

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
					if (!(pointPolygonTest(contours[i],Point(col,row),false) == -1)){
						floodFill(drawing, Point(col,row), color, (cv::Rect*)0, cv::Scalar(), 200);
						break;
					}
				}
			}

		}
	}

	//floodFill(drawing, cv::Point(contours[0][0])+Point(10,10), 255, (cv::Rect*)0, cv::Scalar(), 200);

	imshow( "Result window", drawing );

	//threshold( img2, dstOtsu2, 0, 255, THRESH_OTSU+THRESH_BINARY_INV );
	//imshow("Otsu 2",dstOtsu2);

	/*adaptiveThreshold(img, dstAdaptive, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 69, 1);
	adaptiveThreshold(img2, dstAdaptive2, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 69, 1);
	imshow("Adaptive",dstAdaptive);
	imshow("Adaptive 2", dstAdaptive2);*/
	waitKey(0);

}
