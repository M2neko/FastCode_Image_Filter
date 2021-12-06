#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <immintrin.h>

using namespace std;
using namespace cv;


void nothing(int x, void* data) {}

void tv_60(Mat img) {
	
	namedWindow("image");
	int slider = 0;
	int slider2 = 0;
	createTrackbar("val","image",nullptr,255,nothing);
	setTrackbarPos("val","image",slider);

	createTrackbar("threshold","image",nullptr,100,nothing);
	setTrackbarPos("threshold","image",slider2);

    int height = img.size().height;
	int width = img.size().width;

    // move changed to gray the outer loop to save the time to change it every time in the loop
	Mat gray = img.clone();
    // gray.convertTo(gray,CV_32F);
    Mat channels[3];
    split(gray, channels);
    Mat H = channels[0];
	H.convertTo(H, CV_32F);
	Mat S = channels[1];
	S.convertTo(S, CV_32F);
	Mat V = channels[2];
	V.convertTo(V, CV_32F);
	float* b = &(H.at<float>(0,0));
	float* g = &(S.at<float>(0,0));
	float* r = &(V.at<float>(0,0));
	__m256 calc1 = _mm256_set1_ps(0.299);
	__m256 calc2 = _mm256_set1_ps(0.587);
	__m256 calc3 = _mm256_set1_ps(0.114);

	for (int i=0; i < height * width; i += 8){
		// 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue.
		// int gray_num = 0.299 * r + 0.587 * g + 0.114 * b;
		__m256 b1 = _mm256_load_ps(b + i);
		__m256 g1 = _mm256_load_ps(g + i);
		__m256 r1 = _mm256_load_ps(r + i);

		__m256 b1_mul = _mm256_mul_ps(b1, calc3);
		__m256 g1_mul = _mm256_mul_ps(g1, calc2);
		__m256 r1_mul = _mm256_mul_ps(r1, calc1);

		__m256 gray_num = _mm256_add_ps(b1_mul, g1_mul);
		gray_num = _mm256_add_ps(gray_num, r1_mul);

		_mm256_store_ps(b + i, gray_num);

	}	
    //8 bit - > unsigned char
    //32 bit - > float/ unsigned int;
    gray = H.clone();

	while (true) {
        Mat gray_temp;
        gray_temp = gray.clone();
		float thresh = getTrackbarPos("threshold","image");
		float val = getTrackbarPos("val","image");

		for (int i=0; i < height; i++){
			for (int j=0; j < width; j++){
				if (rand()%100 <= thresh){
					if (rand()%2 == 0)
						gray_temp.at<float>(i,j) = std::min(gray_temp.at<float>(i,j) + (float)(rand()%((int)val+1)), (float)255);
					else
						gray_temp.at<float>(i,j) = std::max(gray_temp.at<float>(i,j) - (float)(rand()%((int)val+1)), (float)0);
				}
			}
		}

    		imshow("original",img);
    		imshow("image",gray_temp);

	    	if (waitKey(1) == 'q')
	    		break;
		}
	destroyAllWindows();
}

int main(){
	Mat img = imread("image.jpg");
	tv_60(img);
	return 0;
}