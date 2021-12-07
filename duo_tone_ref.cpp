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

void nothing(int x, void* data) {}

Mat exponential_function(Mat channel, float exp){
	Mat table(1, 256, CV_32F);

	for (int i = 0; i < 256; i++)
		table.at<float>(i) = (float)min((int)pow(i,exp),255);

	table.convertTo(table, CV_8U);
	channel.convertTo(channel, CV_8U);

	LUT(channel,table,channel);
	return channel;
}

void duo_tone(Mat img, int times){
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

	while(true){
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
		split(img,channels);
		auto start = system_clock::now();
		channels[0].convertTo(channels[0], CV_32F);
		channels[1].convertTo(channels[1], CV_32F);
		channels[2].convertTo(channels[2], CV_32F);
		for (int i=0; i<3; i++){
			if ((i == s1)||(i==s2)){
				channels[i] = exponential_function(channels[i],exp);
			}
			else{
				if (s3){
					channels[i] = exponential_function(channels[i],2-exp);
				}
				else{
					channels[i] = Mat::zeros(channels[i].size(),CV_8UC1);
				}
			}
		}

		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		dsum += duration.count();
		vector<Mat> newChannels{channels[0],channels[1],channels[2]};
		merge(newChannels,res);

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
