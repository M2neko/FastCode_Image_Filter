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

void nothing(int x, void *data) {}

void test() {
	__m256 num1 = _mm256_set1_ps(i);
	__m256 num2 = _mm256_set1_ps(i);




	// for (int i = 0; i < )
}

Mat exponential_function(Mat channel, float exp)
{
	Mat table(1, 256, CV_32F,Scalar(255));
	__m256 e = _mm256_set1_ps(exp);
	float *x = &(table.at<float>(0));

//#pragma omp parallel num_threads(NUM_THREAD)
	if (exp<1.0){
		for (int i = 0; i < 256; i += 8){
			__m256 num = _mm256_set1_ps(i);
			__m256 duo = _mm256_pow_ps(num, e);

			mul 


			_mm256_store_ps(x + i, duo);
		}
	}else{
		__m256 q = _mm256_set1_ps(255.0);
		for (int i = 0; i < 256; i += 8){
			__m256 num = _mm256_set1_ps(i);
			__m256 duo = _mm256_pow_ps(num, e);
			__m256 set = _mm256_min_ps(duo, q);
			_mm256_store_ps(x + i, set);
			if (set[0]==q[0]){
				break;
			}
		}
	}

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

int main(int argc, char** argv)
{
	int times = 1;
	if (!TEST_MODE)
		times = atoi(argv[1]);
	Mat img = imread("image.jpg");
	duo_tone(img, times);
	return 0;
}