#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <immintrin.h>
#include <chrono>
#define TEST_MODE 0

using namespace std;
using namespace cv;
using namespace chrono;

double dsum = 0.0;

void nothing(int x, void *data) {}

void brightness(Mat img, int times)
{
	for (int i = 0; i < times; i++)
	{

		Mat hsv;

		cvtColor(img, hsv, COLOR_BGR2HSV);
		float val = 38;
		if (TEST_MODE) cin >> val;
		val = val / 100.0;

		Mat channels[3];
		split(hsv, channels);
		Mat H = channels[0];
		H.convertTo(H, CV_32F);
		Mat S = channels[1];
		S.convertTo(S, CV_32F);
		Mat V = channels[2];
		V.convertTo(V, CV_32F);

		auto start = system_clock::now();
		__m256 b = _mm256_set1_ps(val);
		__m256 q = _mm256_set1_ps(255.0);
		float *x = &(S.at<float>(0, 0));
		float *y = &(V.at<float>(0, 0));
		for (int i = 0; i < H.size().height * H.size().width; i += 48)
		{
			_mm256_store_ps(x + i, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x + i), b), q));
			_mm256_store_ps(x + i + 8, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x + i + 8), b), q));
			_mm256_store_ps(x + i + 16, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x + i + 16), b), q));
			_mm256_store_ps(x + i + 24, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x + i + 24), b), q));
			_mm256_store_ps(x + i + 32, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x + i + 32), b), q));
			_mm256_store_ps(x + i + 40, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x + i + 40), b), q));

			_mm256_store_ps(y + i, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y + i), b), q));
			_mm256_store_ps(y + i + 8, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y + i + 8), b), q));
			_mm256_store_ps(y + i + 16, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y + i + 16), b), q));
			_mm256_store_ps(y + i + 24, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y + i + 24), b), q));
			_mm256_store_ps(y + i + 32, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y + i + 32), b), q));
			_mm256_store_ps(y + i + 40, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y + i + 40), b), q));
		}

		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		dsum += duration.count();
		H.convertTo(H, CV_8U);
		S.convertTo(S, CV_8U);
		V.convertTo(V, CV_8U);

		vector<Mat> hsvChannels{H, S, V};
		Mat hsvNew;
		merge(hsvChannels, hsvNew);

		Mat res;
		cvtColor(hsvNew, res, COLOR_HSV2BGR);
		if (TEST_MODE) {
			imshow("original",img);
			imshow("image",res);
			waitKey(0);
		}
	}
}

int main(int argc, char **argv)
{
	int times = atoi(argv[1]);
	Mat img = imread("image2.jpg");
	// auto start = system_clock::now();
	brightness(img, times);
	// auto end = system_clock::now();
	// auto duration = duration_cast<microseconds>(end - start);
	// double conTime = dsum;
	// int m = 4;
	// int n = 8;
	// cout << m * n * times / conTime << endl;
	cout << "It takes "
	<< dsum * microseconds::period::num / microseconds::period::den * 1000.0 / double(times)
	<< " milliseconds" << endl;
	return 0;
}
