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
		// cin >> val;
		val = val / 100.0;
		__m128 b = _mm_set1_ps(val);
		__m128 q = _mm_set1_ps(255.0);

		Mat channels[3];
		split(hsv, channels);
		Mat H = channels[0];
		H.convertTo(H, CV_32F);
		Mat S = channels[1];
		S.convertTo(S, CV_32F);
		Mat V = channels[2];
		V.convertTo(V, CV_32F);

		float* x = &(S.at<float>(0, 0));
		float* y = &(V.at<float>(0, 0));

		for (int i = 0; i < H.size().height * H.size().width; i += 4) {
			__m128 a = _mm_set_ps(*(x + i + 3), *(x + i + 2), *(x + i + 1), *(x + i));
			_mm_store_ps(x + i, _mm_min_ps(_mm_mul_ps(a, b), q));

			__m128 d = _mm_set_ps(*(y + i + 3), *(y + i + 2), *(y + i + 1), *(y + i));
			_mm_store_ps(y + i, _mm_min_ps(_mm_mul_ps(d, b), q));
		}

		H.convertTo(H, CV_8U);
		S.convertTo(S, CV_8U);
		V.convertTo(V, CV_8U);

		vector<Mat> hsvChannels{H, S, V};
		Mat hsvNew;
		merge(hsvChannels, hsvNew);

		Mat res;
		cvtColor(hsvNew, res, COLOR_HSV2BGR);

		// imshow("original",img);
    	// imshow("image",res);
		// waitKey(0);
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
