#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <immintrin.h>
#include <omp.h>
#include "avx_mathfun.h"
#define NUM_THREAD 16
#define TEST_MODE 0

using namespace std;
using namespace cv;



inline static __m256 _mm256_pow_ps(__m256 a, __m256 b) {

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
		for (int i = 0; i < 256; i += 8)
		{
			__m256 num = _mm256_set1_ps(i);
			__m256 duo = _mm256_pow_ps(num, e);
			_mm256_store_ps(x + i, duo);
		}
	}
	else
	{
		__m256 q = _mm256_set1_ps(255.0);
		for (int i = 0; i < 256; i += 8)
		{
			__m256 num = _mm256_set_ps(i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7);
			__m256 duo = _mm256_pow_ps(num, e);
			__m256 set = _mm256_min_ps(duo, q);
			_mm256_store_ps(x + i, set);
			if (set[0] == q[0])
			{
				break;
			}
		}
	}

	table.convertTo(table, CV_8U);
	channel.convertTo(channel, CV_8U);

	LUT(channel, table, channel);
	return channel;
}

void duo_tone(Mat img, int times)
{
	for (int x = 0; x < times; x++)
	{
		namedWindow("image");
		int slider1 = 0;
		int slider2 = 1;
		int slider3 = 3;
		int slider4 = 0;
		string switch1 = "0 : BLUE n1 : GREEN n2 : RED";
		string switch2 = "0 : BLUE n1 : GREEN n2 : RED n3 : NONE";
		string switch3 = "0 : DARK n1 : LIGHT";

		createTrackbar("exponent", "image", nullptr, 10, nothing);
		setTrackbarPos("exponent", "image", slider1);

		createTrackbar(switch1, "image", nullptr, 2, nothing);
		setTrackbarPos(switch1, "image", slider2);

		createTrackbar(switch2, "image", nullptr, 3, nothing);
		setTrackbarPos(switch2, "image", slider3);

		createTrackbar(switch3, "image", nullptr, 1, nothing);
		setTrackbarPos(switch3, "image", slider4);

		// createTrackbar("exponent","image",&slider1,10,nothing);
		// createTrackbar(switch1,"image",&slider2,2,nothing);
		// createTrackbar(switch2,"image",&slider3,3,nothing);
		// createTrackbar(switch3,"image",&slider4,1,nothing);
		omp_set_num_threads(3);
		while (true)
		{
			int exp1 = getTrackbarPos("exponent", "image");
			float exp = 1 + exp1 / 100.0;
			int s1 = getTrackbarPos(switch1, "image");
			int s2 = getTrackbarPos(switch2, "image");
			int s3 = getTrackbarPos(switch3, "image");
			Mat res = img.clone();
			Mat channels[3];
			split(img, channels);
			channels[0].convertTo(channels[0], CV_32F);
			channels[1].convertTo(channels[1], CV_32F);
			channels[2].convertTo(channels[2], CV_32F);
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
			vector<Mat> newChannels{channels[0], channels[1], channels[2]};
			merge(newChannels, res);
			imshow("Original", img);
			imshow("image", res);
			if (waitKey(1) == 'q')
				break;
		}
		destroyAllWindows();
	}
}

int main(int argc, char **argv)
{
	int times = 1;
	// if (!TEST_MODE)
	// 	times = atoi(argv[1]);
	Mat img = imread("image2.jpg");
	duo_tone(img, times);
	return 0;

}