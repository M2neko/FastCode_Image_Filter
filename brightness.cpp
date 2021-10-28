#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <immintrin.h>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;

void nothing(int x, void *data) {}

void brightness(Mat img, int times)
{
	// for (int i = 0; i < times; i++)
	// {

		Mat hsv;

		cvtColor(img, hsv, COLOR_BGR2HSV);
		float val = 38;
		// cin >> val;
		val = val / 100.0;
		__m512 b = _mm512_set1_ps(val);
		__m512 q = _mm512_set1_ps(255.0);

	// 	Mat channels[3];
	// 	split(hsv, channels);
	// 	Mat H = channels[0];
	// 	H.convertTo(H, CV_32F);
	// 	Mat S = channels[1];
	// 	S.convertTo(S, CV_32F);
	// 	Mat V = channels[2];
	// 	V.convertTo(V, CV_32F);

	// 	float* x = &(S.at<float>(0, 0));
	// 	float* y = &(V.at<float>(0, 0));


	// 	for (int i = 0; i < H.size().height * H.size().width; i += 16) {
	// 		_mm512_store_ps(x + i, _mm512_min_ps(_mm512_mul_ps(_mm512_load_ps(x + i), b), q));
	// 		_mm512_store_ps(y + i, _mm512_min_ps(_mm512_mul_ps(_mm512_load_ps(y + i), b), q));
	// 	}

	// 	H.convertTo(H, CV_8U);
	// 	S.convertTo(S, CV_8U);
	// 	V.convertTo(V, CV_8U);

	// 	vector<Mat> hsvChannels{H, S, V};
	// 	Mat hsvNew;
	// 	merge(hsvChannels, hsvNew);

	// 	Mat res;
	// 	cvtColor(hsvNew, res, COLOR_HSV2BGR);

	// 	// imshow("original",img);
    // 	// imshow("image",res);
	// 	// waitKey(0);
	// }
}

int main(int argc, char **argv)
{
	int times = atoi(argv[1]);
	Mat img = imread("image2.jpg");
	auto start = system_clock::now();
	brightness(img, times);
	auto end = system_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	cout << "It takes "
		 << double(duration.count()) * microseconds::period::num / microseconds::period::den
		 << " seconds" << endl;
	return 0;
}
