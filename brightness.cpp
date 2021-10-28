#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <xmmintrin.h>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;

void nothing(int x, void *data) {}

void brightness(Mat img, int times)
{
	for (int i = 0; i < times; i++)
	{


		Mat hsv;

		cvtColor(img, hsv, COLOR_BGR2HSV);
		float val = 37;
		val = val / 100.0;
		__m128 b = _mm_set1_ps(val);

		Mat channels[3];
		split(hsv, channels);
		Mat H = channels[0];
		H.convertTo(H, CV_32F);
		Mat S = channels[1];
		S.convertTo(S, CV_32F);
		Mat V = channels[2];
		V.convertTo(V, CV_32F);

		for (int i = 0; i < H.size().height; i++)
		{
			for (int j = 0; j < H.size().width; j += 4)
			{
				__m128 a = _mm_set_ps(S.at<float>(i, j + 3), S.at<float>(i, j + 2), S.at<float>(i, j + 1), S.at<float>(i, j));
				_mm_store_ps(&S.at<float>(i, j), _mm_mul_ps(a, b));

				// S.at<float>(i,j) = min(S.at<float>(i,j), 255);
				// _mm512_gmin_pd (__m512d a, (255.0, 255.0, 255.0, 255.0)))
				if (S.at<float>(i, j) > 255)
					S.at<float>(i, j) = 255;

				if (S.at<float>(i, j + 1) > 255)
					S.at<float>(i, j + 1) = 255;

				if (S.at<float>(i, j + 2) > 255)
					S.at<float>(i, j + 2) = 255;

				if (S.at<float>(i, j + 3) > 255)
					S.at<float>(i, j + 3) = 255;

				// scale pixel values up or down for channel 2(Value)

				__m128 d = _mm_set_ps(V.at<float>(i, j + 3), V.at<float>(i, j + 2), V.at<float>(i, j + 1), V.at<float>(i, j));
				_mm_store_ps(&V.at<float>(i, j), _mm_mul_ps(d, b));

				if (V.at<float>(i, j) > 255)
					V.at<float>(i, j) = 255;

				if (V.at<float>(i, j + 1) > 255)
					V.at<float>(i, j + 1) = 255;

				if (V.at<float>(i, j + 2) > 255)
					V.at<float>(i, j + 2) = 255;

				if (V.at<float>(i, j + 3) > 255)
					V.at<float>(i, j + 3) = 255;
			}
		}

		// for (int i = 0; i < H.size().height; i++) {
		// 	for (int j = 0; j < H.size().width; j += 4) {

		// 		auto a = _mm_set_ps(S.at<float>(i, j), S.at<float>(i, j + 1), S.at<float>(i, j + 2), S.at<float>(i, j + 3));
		// 		auto b = _mm_set1_ps(val);
		// 		auto c =_mm_mul_ps(a, b);
		// 		S.at<float>(i, j) = c[0];
		// 		S.at<float>(i, j + 1) = c[1];
		// 		S.at<float>(i, j + 2) = c[2];
		// 		S.at<float>(i, j + 3) = c[3];

		// 		_mm_store_ps(&S.at<float>(i, j) ,_mm_mul_ps(a, b));

		// _mm_set_ps(S[i][j], S[i][j + 1], S[i][j + 2], S[i][j + 3]);
		// _mm_set1_ps(val);
		// _mm_mul_ps (a, b)

		// 		_mm_set_ps(S[i][j + 4], S[i][j + 5], S[i][j + 6], S[i][j + 7]);
		// 		_mm_set1_ps(val);

		// 		_m256_set_pd(S[i][0], S[i][1], S[i][2], S[i][3]);
		// 		_m256_set_pd(val, 0, 0, 0);

		// 		S[i][j] *= val;
		// 		S[i][j] = S[i][j] > 255 ? 255 : S[i][j];
		// 	}
		// }

		H.convertTo(H, CV_8U);
		S.convertTo(S, CV_8U);
		V.convertTo(V, CV_8U);

		vector<Mat> hsvChannels{H, S, V};
		Mat hsvNew;
		merge(hsvChannels, hsvNew);

		Mat res;
		cvtColor(hsvNew, res, COLOR_HSV2BGR);
	}
}

int main(int argc, char **argv)
{
	int times = atoi(argv[1]);
	Mat img = imread("image.jpg");
	auto start = system_clock::now();
	brightness(img, times);
	auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
	cout << "It takes "
		 << double(duration.count()) * microseconds::period::num / microseconds::period::den
		 << " seconds" << endl;
	return 0;
}
