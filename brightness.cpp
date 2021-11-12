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
			__m256 temp111 = _mm256_load_ps(x + i);
			__m256 temp112 = _mm256_mul_ps(temp111, b);
			__m256 temp113 = _mm256_min_ps(temp112, q);
			_mm256_store_ps(x + i, temp113);

			__m256 temp211 = _mm256_load_ps(y + i);
			__m256 temp212 = _mm256_mul_ps(temp211, b);
			__m256 temp213 = _mm256_min_ps(temp212, q);
			_mm256_store_ps(y + i, temp213);

			if (i + 8 == H.size().height * H.size().width) break;

			__m256 temp121 = _mm256_load_ps(x + i + 8);
			__m256 temp122 = _mm256_mul_ps(temp121, b);
			__m256 temp123 = _mm256_min_ps(temp122, q);
			_mm256_store_ps(x + i + 8, temp123);

			__m256 temp221 = _mm256_load_ps(y + i + 8);
			__m256 temp222 = _mm256_mul_ps(temp221, b);
			__m256 temp223 = _mm256_min_ps(temp222, q);
			_mm256_store_ps(y + i + 8, temp223);

			if (i + 16 == H.size().height * H.size().width) break;

			__m256 temp131 = _mm256_load_ps(x + i + 16);
			__m256 temp132 = _mm256_mul_ps(temp131, b);
			__m256 temp133 = _mm256_min_ps(temp132, q);
			_mm256_store_ps(x + i + 16, temp133);

			__m256 temp231 = _mm256_load_ps(y + i + 16);
			__m256 temp232 = _mm256_mul_ps(temp231, b);
			__m256 temp233 = _mm256_min_ps(temp232, q);
			_mm256_store_ps(y + i + 16, temp233);

			if (i + 24 == H.size().height * H.size().width) break;

			__m256 temp141 = _mm256_load_ps(x + i + 24);
			__m256 temp142 = _mm256_mul_ps(temp141, b);
			__m256 temp143 = _mm256_min_ps(temp142, q);
			_mm256_store_ps(x + i + 24, temp143);

			__m256 temp241 = _mm256_load_ps(y + i + 24);
			__m256 temp242 = _mm256_mul_ps(temp241, b);
			__m256 temp243 = _mm256_min_ps(temp242, q);
			_mm256_store_ps(y + i + 24, temp243);

			if (i + 32 == H.size().height * H.size().width) break;


			__m256 temp151 = _mm256_load_ps(x + i + 32);
			__m256 temp152 = _mm256_mul_ps(temp151, b);
			__m256 temp153 = _mm256_min_ps(temp152, q);
			_mm256_store_ps(x + i + 32, temp153);

			__m256 temp251 = _mm256_load_ps(y + i + 32);
			__m256 temp252 = _mm256_mul_ps(temp251, b);
			__m256 temp253 = _mm256_min_ps(temp252, q);
			_mm256_store_ps(y + i + 32, temp253);

			if (i + 40 == H.size().height * H.size().width) break;

			__m256 temp161 = _mm256_load_ps(x + i + 40);
			__m256 temp162 = _mm256_mul_ps(temp161, b);
			__m256 temp163 = _mm256_min_ps(temp162, q);
			_mm256_store_ps(x + i + 40, temp163);

			__m256 temp261 = _mm256_load_ps(y + i + 40);
			__m256 temp262 = _mm256_mul_ps(temp261, b);
			__m256 temp263 = _mm256_min_ps(temp262, q);
			_mm256_store_ps(y + i + 40, temp263);
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
	Mat img = imread("image5.jpg");
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
