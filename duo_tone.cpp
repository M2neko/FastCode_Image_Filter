#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include "avx_mathfun.h"
#define NUM_THREAD 2
#define TEST_MODE 0

using namespace std;
using namespace cv;
using namespace chrono;

double dsum = 0.0;

inline static __m256 _mm256_pow_ps(__m256 a, __m256 b)
{

	__m256 float_max = _mm256_set1_ps((float)INT_MAX);
	__m256 float_zero = _mm256_set1_ps((float)0.0);
	__m256 log_res = log256_ps(a);
	__m256 mul_res = _mm256_mul_ps(b, log_res);
	__m256 exp_ps = exp256_ps(mul_res);
	__m256 cmp_val = _mm256_cmp_ps(exp_ps, float_max, _CMP_GE_OQ);
	return _mm256_blendv_ps(exp_ps, float_zero, cmp_val);
}

void nothing(int x, void *data) {}

Mat exponential_function(Mat channel, float exp)
{
	Mat table(1, 256, CV_32F, Scalar(255));
	__m256 e = _mm256_set1_ps(exp);
	float *x = &(table.at<float>(0));

	if (exp < 1.0)
	{
		for (int i = 0; i < 256; i += 32)
		{
			__m256 num1 = _mm256_set_ps(i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7);
			__m256 num2 = _mm256_set_ps(i + 8, i + 9, i + 10, i + 11, i + 12, i + 13, i + 14, i + 15);
			__m256 num3 = _mm256_set_ps(i + 16, i + 17, i + 18, i + 19, i + 20, i + 21, i + 22, i + 23);
			__m256 num4 = _mm256_set_ps(i + 24, i + 25, i + 26, i + 27, i + 28, i + 29, i + 30, i + 31);

			__m256 duo1 = _mm256_pow_ps(num1, e);
			__m256 duo2 = _mm256_pow_ps(num2, e);
			__m256 duo3 = _mm256_pow_ps(num3, e);
			__m256 duo4 = _mm256_pow_ps(num4, e);

			_mm256_store_ps(x + i, duo1);
			_mm256_store_ps(x + i + 8, duo2);
			_mm256_store_ps(x + i + 16, duo3);
			_mm256_store_ps(x + i + 24, duo4);
		}
	}
	else
	{
		__m256 q = _mm256_set1_ps(255.0);
		for (int i = 0; i < 256 - 24; i += 24)
		{
			__m256 num1 = _mm256_set_ps(i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7);
			__m256 num2 = _mm256_set_ps(i + 8, i + 9, i + 10, i + 11, i + 12, i + 13, i + 14, i + 15);
			__m256 num3 = _mm256_set_ps(i + 16, i + 17, i + 18, i + 19, i + 20, i + 21, i + 22, i + 23);
			__m256 num4 = _mm256_set_ps(i + 24, i + 25, i + 26, i + 27, i + 28, i + 29, i + 30, i + 31);

			__m256 duo1 = _mm256_pow_ps(num1, e);
			__m256 duo2 = _mm256_pow_ps(num2, e);
			__m256 duo3 = _mm256_pow_ps(num3, e);
			__m256 duo4 = _mm256_pow_ps(num4, e);

			__m256 set1 = _mm256_min_ps(duo1, q);
			__m256 set2 = _mm256_min_ps(duo2, q);
			__m256 set3 = _mm256_min_ps(duo3, q);
			__m256 set4 = _mm256_min_ps(duo4, q);

			_mm256_store_ps(x + i, set1);
			_mm256_store_ps(x + i + 8, set2);
			_mm256_store_ps(x + i + 16, set3);
			_mm256_store_ps(x + i + 24, set4);

			if (fabs(set1[0] - q[0]) <= 0.01f)
			{
				break;
			}
		}
	}

	table.convertTo(table, CV_8U);

	LUT(channel, table, channel);
	return channel;
}

void duo_tone(Mat img, int times)
{
	string switch1 = "0 : BLUE n1 : GREEN n2 : RED";
	string switch2 = "0 : BLUE n1 : GREEN n2 : RED n3 : NONE";
	string switch3 = "0 : DARK n1 : LIGHT";
	if (TEST_MODE)
	{
		namedWindow("image");
		int slider1 = 0;
		int slider2 = 1;
		int slider3 = 3;
		int slider4 = 0;

		createTrackbar("exponent", "image", nullptr, 10, nothing);
		setTrackbarPos("exponent", "image", slider1);

		createTrackbar(switch1, "image", nullptr, 2, nothing);
		setTrackbarPos(switch1, "image", slider2);

		createTrackbar(switch2, "image", nullptr, 3, nothing);
		setTrackbarPos(switch2, "image", slider3);

		createTrackbar(switch3, "image", nullptr, 1, nothing);
		setTrackbarPos(switch3, "image", slider4);
	}

	int exp1;
	float exp;
	int s1;
	int s2;
	int s3;
	int count = 0;

	omp_set_num_threads(3);
	while (true)
	{
		if (TEST_MODE)
		{
			exp1 = getTrackbarPos("exponent", "image");
			exp = 1 + exp1 / 100.0;
			s1 = getTrackbarPos(switch1, "image");
			s2 = getTrackbarPos(switch2, "image");
			s3 = getTrackbarPos(switch3, "image");
		}
		else
		{
			exp1 = 4;
			exp = 1 + exp1 / 100.0;
			s1 = 1;
			s2 = 2;
			s3 = 1;
		}
		Mat res = img.clone();
		Mat channels[3];
		split(img, channels);
		auto start = system_clock::now();

#pragma omp parallel
		{
			unsigned int id = omp_get_thread_num();
			if ((id == s1) || (id == s2))
			{
				channels[id] = exponential_function(channels[id], exp);
			}
			else
			{
				if (s3)
				{
					channels[id] = exponential_function(channels[id], 2 - exp);
				}
				else
				{
					channels[id] = Mat::zeros(channels[id].size(), CV_8UC1);
				}
			}
		}

		/* Uncomment if you want to use UNROLLING */

		// if ((s1 == 0) || (s2 == 0))
		// {
		// 	channels[0] = exponential_function(channels[0], exp);
		// }
		// else
		// {
		// 	if (s3)
		// 	{
		// 		channels[0] = exponential_function(channels[0], 2 - exp);
		// 	}
		// 	else
		// 	{
		// 		channels[0] = Mat::zeros(channels[0].size(), CV_8UC1);
		// 	}
		// }
		// if ((s1 == 1) || (s2 == 1))
		// {
		// 	channels[1] = exponential_function(channels[1], exp);
		// }
		// else
		// {
		// 	if (s3)
		// 	{
		// 		channels[1] = exponential_function(channels[1], 2 - exp);
		// 	}
		// 	else
		// 	{
		// 		channels[1] = Mat::zeros(channels[1].size(), CV_8UC1);
		// 	}
		// }
		// if ((s1 == 2) || (s2 == 2))
		// {
		// 	channels[2] = exponential_function(channels[2], exp);
		// }
		// else
		// {
		// 	if (s3)
		// 	{
		// 		channels[2] = exponential_function(channels[2], 2 - exp);
		// 	}
		// 	else
		// 	{
		// 		channels[2] = Mat::zeros(channels[2].size(), CV_8UC1);
		// 	}
		// }
		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		dsum += duration.count();
		vector<Mat> newChannels{channels[0], channels[1], channels[2]};
		merge(newChannels, res);

		if (!TEST_MODE)
		{
			if (++count > times)
				break;
		}

		if (TEST_MODE)
		{
			imshow("Original", img);
			imshow("image", res);
			if (waitKey(1) == 'q')
				break;
		}
	}

	if (TEST_MODE)
		destroyAllWindows();
}

int main(int argc, char **argv)
{
	int times = 1;

	if (!TEST_MODE)
		times = atoi(argv[1]);

	Mat img = imread("image2.jpg");
	duo_tone(img, times);
	cout << "It takes "
		 << dsum * microseconds::period::num / microseconds::period::den * 1000.0 / double(times)
		 << " milliseconds" << endl;
	return 0;
}