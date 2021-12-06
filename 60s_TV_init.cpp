#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <immintrin.h>
#include <chrono>
#include <omp.h>
#define NUM_THREAD 32
#define TEST_MODE 0

using namespace std;
using namespace cv;
using namespace chrono;

double dsum = 0.0;

static __m128i cur_seed;

inline void rand_sse(unsigned int *result)
{
    __m128i cur_seed_split;
    __m128i multiplier;
    __m128i adder;
    __m128i mod_mask;
    __m128i sra_mask;
    __m128i sseresult;
    static const unsigned int mult[4] = {214013, 17405, 214013, 69069};
    static const unsigned int gadd[4] = {2531011, 10395331, 13737667, 1};
    static const unsigned int mask[4] = {0xFFFFFFFF, 0, 0xFFFFFFFF, 0};
    static const unsigned int masklo[4] = {0x00007FFF, 0x00007FFF, 0x00007FFF, 0x00007FFF};

    adder = _mm_load_si128((__m128i *)gadd);
    multiplier = _mm_load_si128((__m128i *)mult);
    mod_mask = _mm_load_si128((__m128i *)mask);
    sra_mask = _mm_load_si128((__m128i *)masklo);
    cur_seed_split = _mm_shuffle_epi32(cur_seed, _MM_SHUFFLE(2, 3, 0, 1));

    cur_seed = _mm_mul_epu32(cur_seed, multiplier);
    multiplier = _mm_shuffle_epi32(multiplier, _MM_SHUFFLE(2, 3, 0, 1));
    cur_seed_split = _mm_mul_epu32(cur_seed_split, multiplier);

    cur_seed = _mm_and_si128(cur_seed, mod_mask);
    cur_seed_split = _mm_and_si128(cur_seed_split, mod_mask);
    cur_seed_split = _mm_shuffle_epi32(cur_seed_split, _MM_SHUFFLE(2, 3, 0, 1));
    cur_seed = _mm_or_si128(cur_seed, cur_seed_split);
    cur_seed = _mm_add_epi32(cur_seed, adder);

    _mm_storeu_si128((__m128i *)result, cur_seed);
    return;
}

inline void get_rand(unsigned int *result) {
    rand_sse(result);
    rand_sse(result + 4);
}

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

    int height = img.size().height;
    int width = img.size().width;

    auto start = system_clock::now();
    // move changed to gray the outer loop to save the time to change it every time in the loop
    Mat gray = img.clone();
    // gray.convertTo(gray,CV_32F);
    Mat channels[3];
    split(gray, channels);
    Mat H = channels[0];
    H.convertTo(H, CV_32F);
    Mat S = channels[1];
    S.convertTo(S, CV_32F);
    Mat V = channels[2];
    V.convertTo(V, CV_32F);
    float *b = &(H.at<float>(0, 0));
    float *g = &(S.at<float>(0, 0));
    float *r = &(V.at<float>(0, 0));
    __m256 calc1 = _mm256_set1_ps(0.299);
    __m256 calc2 = _mm256_set1_ps(0.587);
    __m256 calc3 = _mm256_set1_ps(0.114);
#pragma omp parallel for num_threads(NUM_THREAD)
    for (int i = 0; i < height * width; i += 8)
    {
        // 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue.
        // int gray_num = 0.299 * r + 0.587 * g + 0.114 * b;
        __m256 b1 = _mm256_load_ps(b + i);
        __m256 g1 = _mm256_load_ps(g + i);
        __m256 r1 = _mm256_load_ps(r + i);

        __m256 b1_mul = _mm256_mul_ps(b1, calc3);
        __m256 g1_mul = _mm256_mul_ps(g1, calc2);
        __m256 r1_mul = _mm256_mul_ps(r1, calc1);

        __m256 gray_num = _mm256_add_ps(b1_mul, g1_mul);
        gray_num = _mm256_add_ps(gray_num, r1_mul);

        _mm256_store_ps(b + i, gray_num);
    }

    gray = H.clone();

    __m256 comp_val1 = _mm256_set1_ps(255);
    __m256 comp_val2 = _mm256_set1_ps(0);
    __m256 set_0 = _mm256_set1_ps(0);
    unsigned int randset[8];

    float thresh = 0.0;
	float val = 0.0;
	int count = 0;

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
	dsum += duration.count();

    while (true)
    {
        start = system_clock::now();
        Mat gray_temp;
        gray_temp = gray.clone();
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
        __m256 get_thresh = _mm256_set1_ps(thresh);
        float *gray_pixel = &(gray_temp.at<float>(0, 0));

#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < height * width; i += 8)
        {
            __m256 gray_channel = _mm256_load_ps(gray_pixel + i);
            get_rand((unsigned int *)randset);
            __m256 rand_val1 = _mm256_set_ps((float)(randset[0] % 100), (float)(randset[1] % 100), (float)(randset[2] % 100), (float)(randset[3] % 100), (float)(randset[4] % 100), (float)(randset[5] % 100), (float)(randset[6] % 100), (float)(randset[7] % 100));
            get_rand((unsigned int *)randset);
            __m256 rand_val2 = _mm256_set_ps((float)(randset[0] % 2), (float)(randset[1] % 2), (float)(randset[2] % 2), (float)(randset[3] % 2), (float)(randset[4] % 2), (float)(randset[5] % 2), (float)(randset[6] % 2), (float)(randset[7] % 2));
            // compare rand() % 100 <= thresh
            // rand_val1[0] - get_thresh[0]是负数，返回0
            // rand_val1[0] - get_thresh[0]是正数，返回nan
            __m256 cmp_val1 = _mm256_cmp_ps(rand_val1, get_thresh, _CMP_GT_OQ);
            // compare rand() % 2 == 0
            // rand_val2[0]是1，返回0
            // rand_val2[0]是0，返回nan
            __m256 cmp_val2 = _mm256_cmp_ps(rand_val2, comp_val2, _CMP_EQ_OQ);
            get_rand((unsigned int *)randset);
            __m256 calc_pos = _mm256_set_ps((float)(randset[0] % ((int)val + 1)), (float)(randset[1] % ((int)val + 1)), (float)(randset[2] % ((int)val + 1)), (float)(randset[3] % ((int)val + 1)), (float)(randset[4] % ((int)val + 1)), (float)(randset[5] % ((int)val + 1)), (float)(randset[6] % ((int)val + 1)), (float)(randset[7] % ((int)val + 1)));
            get_rand((unsigned int *)randset);
            __m256 calc_neg = _mm256_set_ps((float)-(randset[0] % ((int)val + 1)), (float)(-randset[1] % ((int)val + 1)), (float)(-randset[2] % ((int)val + 1)), (float)(-randset[3] % ((int)val + 1)), (float)(-randset[4] % ((int)val + 1)), (float)(-randset[5] % ((int)val + 1)), (float)(-randset[6] % ((int)val + 1)), (float)(-randset[7] % ((int)val + 1)));

            __m256 blend_1 = _mm256_blendv_ps(calc_pos, calc_neg, cmp_val2);
            __m256 add_val1 = _mm256_add_ps(gray_channel, blend_1);
            __m256 blend_2 = _mm256_blendv_ps(add_val1, gray_channel, cmp_val1);

            __m256 bound_1 = _mm256_min_ps(blend_2, comp_val1);
            __m256 bound_2 = _mm256_max_ps(bound_1, comp_val2);

            _mm256_store_ps(gray_pixel + i, bound_2);
        }


        gray_temp.convertTo(gray_temp,CV_8U);

        end = system_clock::now();
        duration = duration_cast<microseconds>(end - start);
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
			imshow("image", gray_temp);
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
	Mat img = imread("image.jpg");
	tv_60(img, times);

	cout << "It takes "
		 << dsum * microseconds::period::num / microseconds::period::den * 1000.0 / double(times)
		 << " milliseconds" << endl;
	return 0;
}