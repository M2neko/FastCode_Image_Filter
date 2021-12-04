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

__m256 b;
__m256 q;

void recursion(int size, float* x, float* y)
{

	if (size == 32)
	{
		_mm256_store_ps(x, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x), b), q));
		_mm256_store_ps(x + 8, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x + 8), b), q));

		_mm256_store_ps(x + 16, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x + 16), b), q));
		_mm256_store_ps(x + 24, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x + 24), b), q));

		_mm256_store_ps(y, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y), b), q));
		_mm256_store_ps(y + 8, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y + 8), b), q));

		_mm256_store_ps(y + 16, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y + 16), b), q));
		_mm256_store_ps(y + 24, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y + 24), b), q));
		return;
	}

	recursion(size / 2, x, y);
	recursion(size / 2, x + size / 2, y + size / 2);

}

void nothing(int x, void *data) {}

void brightness(Mat img, int times)
{
	for (int i = 0; i < times; i++)
	{

		Mat hsv;

		cvtColor(img, hsv, COLOR_BGR2HSV);
		float val = 38;
		if (TEST_MODE)
			cin >> val;
		val = val / 100.0;
		b = _mm256_set1_ps(val);
		q = _mm256_set1_ps(255.0);

		Mat channels[3];
		split(hsv, channels);
		Mat H = channels[0];
		H.convertTo(H, CV_32F);
		Mat S = channels[1];
		S.convertTo(S, CV_32F);
		Mat V = channels[2];
		V.convertTo(V, CV_32F);

		float *x = &(S.at<float>(0, 0));
		float *y = &(V.at<float>(0, 0));

		// for (int i = 0; i < H.size().height * H.size().width; i += 32)
		// {
		// 	_mm256_store_ps(x, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x), b), q));
		// 	_mm256_store_ps(x + 8, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x + 8), b), q));

		// 	_mm256_store_ps(x + 16, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x + 16), b), q));
		// 	_mm256_store_ps(x + 24, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(x + 24), b), q));

		// 	_mm256_store_ps(y, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y), b), q));
		// 	_mm256_store_ps(y + 8, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y + 8), b), q));

		// 	_mm256_store_ps(y + 16, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y + 16), b), q));
		// 	_mm256_store_ps(y + 24, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(y + 24), b), q));
		// }
	
		recursion((int)(H.size().height) * (int)(H.size().width), x, y);

		H.convertTo(H, CV_8U);
		S.convertTo(S, CV_8U);
		V.convertTo(V, CV_8U);

		vector<Mat> hsvChannels{H, S, V};
		Mat hsvNew;
		merge(hsvChannels, hsvNew);

		Mat res;
		cvtColor(hsvNew, res, COLOR_HSV2BGR);
		if (TEST_MODE)
		{
			imshow("original", img);
			imshow("image", res);
			waitKey(0);
		}
	}
}

int main(int argc, char **argv)
{
	int times = atoi(argv[1]);
	Mat img = imread("image4.jpg");
	auto start = system_clock::now();
	brightness(img, times);
	auto end = system_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	cout << "It takes "
		 << double(duration.count()) * microseconds::period::num / microseconds::period::den * 1000.0 / double(times)
		 << " milliseconds" << endl;
	return 0;
}
