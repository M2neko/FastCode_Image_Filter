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

void tv_60(Mat img, int times)
{
	if (TEST_MODE)
	{
		namedWindow("image");
		int slider = 0;
		int slider2 = 0;
		createTrackbar("val", "image", nullptr, 255, nothing);
		setTrackbarPos("val", "image", slider);

		createTrackbar("threshold", "image", nullptr, 100, nothing);
		setTrackbarPos("threshold", "image", slider2);
	}
	float thresh = 0.0;
	float val = 0.0;
	int count = 0;
	while (true)
	{
		auto start = system_clock::now();
		int height = img.size().height;
		int width = img.size().width;
		Mat gray;
		cvtColor(img, gray, COLOR_BGR2GRAY);
		if (TEST_MODE)
		{
			thresh = getTrackbarPos("threshold", "image");
			val = getTrackbarPos("val", "image");
		}
		else
		{
			thresh = 60.5;
			val = 120;
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (rand() % 100 <= thresh)
				{
					if (rand() % 2 == 0)
						gray.at<uchar>(i, j) = std::min(gray.at<uchar>(i, j) + rand() % ((int)val + 1), 255);
					else
						gray.at<uchar>(i, j) = std::max(gray.at<uchar>(i, j) - rand() % ((int)val + 1), 0);
				}
			}
		}

		auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
		dsum += duration.count();

		if (!TEST_MODE)
		{
			if (++count > times)
			{
				break;
			}
		}

		if (TEST_MODE)
		{
			imshow("original", img);
			imshow("image", gray);

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
	times = atoi(argv[1]);
	Mat img = imread("image2.jpg");
	tv_60(img, times);

	cout << "It takes "
		 << dsum * microseconds::period::num / microseconds::period::den * 1000.0 / double(times)
		 << " milliseconds" << endl;
	return 0;
}