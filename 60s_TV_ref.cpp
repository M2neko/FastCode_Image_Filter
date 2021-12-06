#include <iostream>
#include <string>
#include <immintrin.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

static __m128i cur_seed;

void nothing(int x, void* data) {}

inline void rand_sse( unsigned int* result ) {
    __m128i cur_seed_split;
    __m128i multiplier;
    __m128i adder;
    __m128i mod_mask;
    __m128i sra_mask;
    __m128i sseresult;
    static const unsigned int mult[4] = { 214013, 17405, 214013, 69069 };
    static const unsigned int gadd[4] = { 2531011, 10395331, 13737667, 1 };
    static const unsigned int mask[4] = { 0xFFFFFFFF, 0, 0xFFFFFFFF, 0 };
    static const unsigned int masklo[4] = { 0x00007FFF, 0x00007FFF, 0x00007FFF, 0x00007FFF };

    adder = _mm_load_si128( (__m128i*) gadd);
    multiplier = _mm_load_si128( (__m128i*) mult);
    mod_mask = _mm_load_si128( (__m128i*) mask);
    sra_mask = _mm_load_si128( (__m128i*) masklo);
    cur_seed_split = _mm_shuffle_epi32( cur_seed, _MM_SHUFFLE( 2, 3, 0, 1 ) );

    cur_seed = _mm_mul_epu32( cur_seed, multiplier );
    multiplier = _mm_shuffle_epi32( multiplier, _MM_SHUFFLE( 2, 3, 0, 1 ) );
    cur_seed_split = _mm_mul_epu32( cur_seed_split, multiplier );

    cur_seed = _mm_and_si128( cur_seed, mod_mask);
    cur_seed_split = _mm_and_si128( cur_seed_split, mod_mask );
    cur_seed_split = _mm_shuffle_epi32( cur_seed_split, _MM_SHUFFLE( 2, 3, 0, 1 ) );
    cur_seed = _mm_or_si128( cur_seed, cur_seed_split );
    cur_seed = _mm_add_epi32( cur_seed, adder);

    _mm_storeu_si128( (__m128i*) result, cur_seed);
    return;
}

void tv_60(Mat img) {
	
	namedWindow("image");
	int slider = 0;
	int slider2 = 0;
	createTrackbar("val","image",nullptr,255,nothing);
	setTrackbarPos("val","image",slider);

	createTrackbar("threshold","image",nullptr,100,nothing);
	setTrackbarPos("threshold","image",slider2);

    Mat gray;


	while (true) {
		int height = img.size().height;
		int width = img.size().width;
		Mat gray;
		cvtColor(img, gray, COLOR_BGR2GRAY);
        gray.convertTo(gray,CV_32F);
		float thresh = getTrackbarPos("threshold","image");
		float val = getTrackbarPos("val","image");

		for (int i=0; i < height; i++){
			for (int j=0; j < width; j++){
                
                
				if (rand()%100 <= thresh){
					if (rand()%2 == 0)
						gray.at<float>(i,j) = std::min(gray.at<float>(i,j) + (float)(rand()%((int)val+1)), (float)255);




					else
						gray.at<float>(i,j) = std::max(gray.at<float>(i,j) - (float)(rand()%((int)val+1)), (float)0);
				}
			}
		}

    		imshow("original",img);
    		imshow("image",gray);

	    	if (waitKey(1) == 'q')
	    		break;
		}
	destroyAllWindows();
}

int main(){
    // unsigned int res[8];
    // rand_sse(res);
    // rand_sse(res + 4);
    // cout << res[0] << '\n' << res[1] << '\n';
	Mat img = imread("image.jpg");
	tv_60(img);
	return 0;
}