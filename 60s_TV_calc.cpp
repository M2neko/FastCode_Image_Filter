#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define TEST_MODE 0

using namespace std;
using namespace cv;

double dsum = 0.0;

void nothing(int x, void* data) {}

void tv_60(Mat img) {
	
	namedWindow("image2");
	int slider = 0;
	int slider2 = 0;
	createTrackbar("val","image",&slider,255,nothing);
	createTrackbar("threshold","image",&slider2,100,nothing);

	auto start = system_clock::now();
	int count = 0;
	
	while (true) {
		int height = img.size().height;
		int width = img.size().width;
		Mat gray = img.clone();
		// cvtColor(img, gray, COLOR_BGR2GRAY);
		float thresh = getTrackbarPos("threshold","image");
		float val = getTrackbarPos("val","image");

		for (int i=0; i < height; i++){
			for (int j=0; j < width; j++){

				float b = img.at<cv::Vec3b>(i,j)[0];
				float g = img.at<cv::Vec3b>(i,j)[1];
				float r = img.at<cv::Vec3b>(i,j)[2];

				// 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue.
				int gray_num = 0.299 * r + 0.587 * g + 0.114 * b;

				if (rand()%100 <= thresh){
					if (rand()%2 == 0)
						// gray.at<uchar>(i,j) = std::min(gray.at<uchar>(i,j) + rand()%((int)val+1), 255);
						int change_gray =std::min(gray_num + rand()%((int)val+1), 255);

						img.at<cv::Vec3b>(i,j)[0] = change_gray;
						img.at<cv::Vec3b>(i,j)[1] = change_gray;
						img.at<cv::Vec3b>(i,j)[2] = change_gray;

				}else{
						//gray.at<uchar>(i,j) = std::max(gray.at<uchar>(i,j) - rand()%((int)val+1), 0);

						int change_gray =std::max(gray_num - rand()%((int)val+1), 0);

						img.at<cv::Vec3b>(i,j)[0] = change_gray;
						img.at<cv::Vec3b>(i,j)[1] = change_gray;
						img.at<cv::Vec3b>(i,j)[2] = change_gray;
				
				}
			}
		}

		count++;

		if (count >= 1000)
			break
		
		if (TEST_MODE)
		{
			imshow("original",img);
			imshow("image",gray);
			waitKey(0);
		}
	}
	auto end = system_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	dsum += duration.count();
	destroyAllWindows();
}

int main(){
	Mat img = imread("image2.jpg");
	tv_60(img);

	cout << "It takes "
		 << dsum * microseconds::period::num / microseconds::period::den * 1000.0 / 1000
		 << " milliseconds" << endl;
	return 0;
}
