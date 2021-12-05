#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <immintrin.h>
#include <omp.h>
#define NUM_THREAD 16
#define TEST_MODE 0

using namespace std;
using namespace cv;


void nothing(int x, void* data) {}

void tv_60(Mat img) {
	
	namedWindow("image");
	int slider = 0;
	int slider2 = 0;
	createTrackbar("val","image",&slider,255,nothing);
	createTrackbar("threshold","image",&slider2,100,nothing);

	
	
	#pragma omp parallel for num_threads(NUM_THREAD)
	while (true) {
		int height = img.size().height;
		int width = img.size().width;
		Mat gray;
		cvtColor(img, gray, COLOR_BGR2GRAY);
		float thresh = getTrackbarPos("threshold","image");
		float val = getTrackbarPos("val","image");

		float pixel = gray.at<uchar>(0,0);
		__m256 get_thresh = _mm256_set1_ps(thresh);
		__m256 get_val = _mm256_set1_ps((val+1));
		__m256 comp_val1 = _mm256_set1_ps(255);
		__m256 comp_val2 = _mm256_set1_ps(0);

		for (int i = 0; i < height * width; i += 8) {
			__m256 gray_pixel1 = _mm256_load_ps(pixel + i);
			__m256d rand_val1 = _mm256_set_ps(rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand());
			
			if (rand() % 100 <= thresh) {

				if (rand() % 100 == 0) {
					// gray.at<uchar>(i,j) = std::min(gray.at<uchar>(i,j) + rand()%((int)val+1), 255);
	
					// rand % val = rand - ((rand / val)*val)
					// __m256i _mm256_rem_epi32 (__m256i a, __m256i b) return a sequence
					// __m256 rem = _mm256_rem_epi32(rand_val1, get_val);
					__m256 right_val1 = _mm256_sub_ps(rand_val1, _mm256_mul_ps(_mm256_div_ps(rand_val1, get_val), get_val));
					__m256 add_val1 = _mm256_add_ps(gray_pixel1, right_val1);
					__m256 min_val1 = _mm256_min_ps(add_val1, comp_val1);
				
				} else {
					//gray.at<uchar>(i,j) = std::max(gray.at<uchar>(i,j) - rand()%((int)val+1), 0);
					
					// __m256i _mm256_rem_epi32 (__m256i a, __m256i b) return a sequence
					__m256 right_val1 = _mm256_sub_ps(rand_val1, _mm256_mul_ps(_mm256_div_ps(rand_val1, get_val), get_val));
					__m256 sub_val1 = _mm256_sub_ps(gray_pixel1, right_val1);
					__m256 max_val1 = _mm256_max_ps(sub_val1, comp_val2);
					
				}
			}
			_mm256_store_ps(pixel + i, max_val1);
		}


    		imshow("original",img);
    		imshow("image",gray);

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
