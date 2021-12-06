#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#define NUM_THREAD 16
#define TEST_MODE 0

using namespace std;
using namespace cv;

double dsum = 0.0;

void nothing(int x, void* data) {}

void tv_60(Mat img) {
	
	namedWindow("image2");
	int slider = 0;
	int slider2 = 0;
	createTrackbar("val","image",&slider,255,nothing);
	createTrackbar("threshold","image",&slider2,100,nothing);

	int count = 0;

	// move changed to gray the outer loop to save the time to change it every time in the loop
	Mat gray = img.clone();
	float* b = &(gray.at<cv::Vec3b>(0,0)[0]);
	float* g = &(gray.at<cv::Vec3b>(0,0)[1]);
	float* r = &(gray.at<cv::Vec3b>(0,0)[2]);
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
		__m256 gray_num = _mm256_add_ps(gray_num, r1_mul);

		_mm256_store_ps(b + i, gray_num);
		_mm256_store_ps(g + i, gray_num);
		_mm256_store_ps(r + i, gray_num);

	}	
	
	while (count <= 1000) {
		int height = img.size().height;
		int width = img.size().width;

		// cvtColor(img, gray, COLOR_BGR2GRAY);
		float thresh = getTrackbarPos("threshold","image");
		float val = getTrackbarPos("val","image");
		__m256 get_thresh = _mm256_set1_ps(thresh);
		__m256 comp_val1 = _mm256_set1_ps(255);
		__m256 comp_val2 = _mm256_set1_ps(0);

		for (int i=0; i < height * width; i += 8){

				__m256 b1 = _mm256_load_ps(b + i);
				__m256 g1 = _mm256_load_ps(g + i);
				__m256 r1 = _mm256_load_ps(r + i);

				__m256 gray_num = _mm256_add_ps(b1, g1);
				__m256 gray_num = _mm256_add_ps(gray_num, r1);

			    __m256 rand_val1 = _mm256_set_ps((float)rand()%100, (float)rand()%100, (float)rand()%100, (float)rand()%100, (float)rand()%100, (float)rand()%100, (float)rand()%100, (float)rand()%100);
				__m256 rand_val2 = _mm256_set_ps((float)rand()%2, (float)rand()%2, (float)rand()%2 (float)rand()%2, (float)rand()%2, (float)rand()%2, (float)rand()%2, (float)rand()%2);
				

				0 0xfffff
				_m256_and


				// if rand_val1 <= get_thresh
				if (rand()%100 <= thresh){
					if (rand()%2 == 0)
						// gray.at<uchar>(i,j) = std::min(gray.at<uchar>(i,j) + rand()%((int)val+1), 255);
						// int change_gray =std::min(gray_num + rand()%((int)val+1), 255);
						__m256 calc1 = _mm256_set_ps((float)rand()%(val+1), (float)rand()%(val+1), (float)rand()%(val+1), (float)rand()%(val+1), (float)rand()%(val+1), (float)rand()%(val+1), (float)rand()%(val+1), (float)rand()%(val+1));
						__m256 add_val1 = _mm256_add_ps(gray_num, calc1);
						__m256 min_val1 = _mm256_min_ps(add_val1, comp_val1);

						_mm256_store_ps(b + i, min_val1)

				}else{
						//gray.at<uchar>(i,j) = std::max(gray.at<uchar>(i,j) - rand()%((int)val+1), 0);

						int change_gray =std::max(gray_num - rand()%((int)val+1), 0);

						img.at<cv::Vec3b>(i,j)[0] = change_gray;
						img.at<cv::Vec3b>(i,j)[1] = change_gray;
						img.at<cv::Vec3b>(i,j)[2] = change_gray;
				
				}
			}
		}

		count++;
		if (TEST_MODE)
		{
			imshow("original",img);
			imshow("image",gray);
			waitKey(0);
		}
	}
	auto end = system_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	dsum += duration.count();
	destroyAllWindows();
}

int main(){
	Mat img = imread("image2.jpg");
	
	tv_60(img);

	if (!TEST_MODE)
	{
		cout << "It takes "
			<< dsum * microseconds::period::num / microseconds::period::den * 1000.0 / 1000
			<< " milliseconds" << endl;
	}
	return 0;
}
