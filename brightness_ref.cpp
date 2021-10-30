#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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

		for (int i = 0; i < H.size().height; i++)
		{
			for (int j = 0; j < H.size().width; j++)
			{
				// scale pixel values up or down for channel 1(Saturation)
				S.at<float>(i, j) *= val;
				if (S.at<float>(i, j) > 255)
					S.at<float>(i, j) = 255;

				// scale pixel values up or down for channel 2(Value)
				V.at<float>(i, j) *= val;
				if (V.at<float>(i, j) > 255)
					V.at<float>(i, j) = 255;
			}
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
	cout << "It takes "
		 << dsum * microseconds::period::num / microseconds::period::den * 1000.0 / double(times)
		 << " milliseconds" << endl;
	return 0;
}
