#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <immintrin.h>
#include <chrono>
#include <omp.h>
#define NUM_THREAD 4
#define TEST_MODE 0

using namespace std;
using namespace cv;
using namespace chrono;

double dsum = 0.0;

static __m128i cur_seed;

// Reference: https://stackoverflow.com/questions/1640258/need-a-fast-random-generator-for-c
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
        for (int i = 0; i < height * width; i += 40)
        {
            __m256 gray_channel1 = _mm256_load_ps(gray_pixel + i);
            get_rand((unsigned int *)randset);
            __m256 rand_val11 = _mm256_set_ps((float)(randset[0] % 100), (float)(randset[1] % 100), (float)(randset[2] % 100), (float)(randset[3] % 100), (float)(randset[4] % 100), (float)(randset[5] % 100), (float)(randset[6] % 100), (float)(randset[7] % 100));
            get_rand((unsigned int *)randset);
            __m256 rand_val12 = _mm256_set_ps((float)(randset[0] % 2), (float)(randset[1] % 2), (float)(randset[2] % 2), (float)(randset[3] % 2), (float)(randset[4] % 2), (float)(randset[5] % 2), (float)(randset[6] % 2), (float)(randset[7] % 2));
            // compare rand() % 100 <= thresh
            // rand_val111[0] - get_thresh[0] negative, return 0
            // rand_val11[0] - get_thresh[0] positive, return NAN
            __m256 cmp_val11 = _mm256_cmp_ps(rand_val11, get_thresh, _CMP_GT_OQ);
            // compare rand() % 2 == 0
            // rand_val12[0] is 1, return 0
            // rand_val12[0] is 0, return NAN
            __m256 cmp_val12 = _mm256_cmp_ps(rand_val12, comp_val2, _CMP_EQ_OQ);
            get_rand((unsigned int *)randset);
            __m256 calc_pos1 = _mm256_set_ps((float)(randset[0] % ((int)val + 1)), (float)(randset[1] % ((int)val + 1)), (float)(randset[2] % ((int)val + 1)), (float)(randset[3] % ((int)val + 1)), (float)(randset[4] % ((int)val + 1)), (float)(randset[5] % ((int)val + 1)), (float)(randset[6] % ((int)val + 1)), (float)(randset[7] % ((int)val + 1)));
            get_rand((unsigned int *)randset);
            __m256 calc_neg1 = _mm256_set_ps((float)-(randset[0] % ((int)val + 1)), (float)(-randset[1] % ((int)val + 1)), (float)(-randset[2] % ((int)val + 1)), (float)(-randset[3] % ((int)val + 1)), (float)(-randset[4] % ((int)val + 1)), (float)(-randset[5] % ((int)val + 1)), (float)(-randset[6] % ((int)val + 1)), (float)(-randset[7] % ((int)val + 1)));

            __m256 blend_11 = _mm256_blendv_ps(calc_pos1, calc_neg1, cmp_val12);
            __m256 add_val11 = _mm256_add_ps(gray_channel1, blend_11);
            __m256 blend_12 = _mm256_blendv_ps(add_val11, gray_channel1, cmp_val11);

            __m256 bound_11 = _mm256_min_ps(blend_12, comp_val1);
            __m256 bound_12 = _mm256_max_ps(bound_11, comp_val2);

            _mm256_store_ps(gray_pixel + i, bound_12);




            __m256 gray_channel2 = _mm256_load_ps(gray_pixel + i + 8);
            get_rand((unsigned int *)randset);
            __m256 rand_val21 = _mm256_set_ps((float)(randset[0] % 100), (float)(randset[1] % 100), (float)(randset[2] % 100), (float)(randset[3] % 100), (float)(randset[4] % 100), (float)(randset[5] % 100), (float)(randset[6] % 100), (float)(randset[7] % 100));
            get_rand((unsigned int *)randset);
            __m256 rand_val22 = _mm256_set_ps((float)(randset[0] % 2), (float)(randset[1] % 2), (float)(randset[2] % 2), (float)(randset[3] % 2), (float)(randset[4] % 2), (float)(randset[5] % 2), (float)(randset[6] % 2), (float)(randset[7] % 2));
            __m256 cmp_val21 = _mm256_cmp_ps(rand_val21, get_thresh, _CMP_GT_OQ);
            __m256 cmp_val22 = _mm256_cmp_ps(rand_val22, comp_val2, _CMP_EQ_OQ);
            get_rand((unsigned int *)randset);
            __m256 calc_pos2 = _mm256_set_ps((float)(randset[0] % ((int)val + 1)), (float)(randset[1] % ((int)val + 1)), (float)(randset[2] % ((int)val + 1)), (float)(randset[3] % ((int)val + 1)), (float)(randset[4] % ((int)val + 1)), (float)(randset[5] % ((int)val + 1)), (float)(randset[6] % ((int)val + 1)), (float)(randset[7] % ((int)val + 1)));
            get_rand((unsigned int *)randset);
            __m256 calc_neg2 = _mm256_set_ps((float)-(randset[0] % ((int)val + 1)), (float)(-randset[1] % ((int)val + 1)), (float)(-randset[2] % ((int)val + 1)), (float)(-randset[3] % ((int)val + 1)), (float)(-randset[4] % ((int)val + 1)), (float)(-randset[5] % ((int)val + 1)), (float)(-randset[6] % ((int)val + 1)), (float)(-randset[7] % ((int)val + 1)));

            __m256 blend_21 = _mm256_blendv_ps(calc_pos2, calc_neg2, cmp_val22);
            __m256 add_val21 = _mm256_add_ps(gray_channel2, blend_21);
            __m256 blend_22 = _mm256_blendv_ps(add_val21, gray_channel2, cmp_val21);

            __m256 bound_21 = _mm256_min_ps(blend_22, comp_val1);
            __m256 bound_22 = _mm256_max_ps(bound_21, comp_val2);

            _mm256_store_ps(gray_pixel + i + 8, bound_22);




            __m256 gray_channel3 = _mm256_load_ps(gray_pixel + i + 16);
            get_rand((unsigned int *)randset);
            __m256 rand_val31 = _mm256_set_ps((float)(randset[0] % 100), (float)(randset[1] % 100), (float)(randset[2] % 100), (float)(randset[3] % 100), (float)(randset[4] % 100), (float)(randset[5] % 100), (float)(randset[6] % 100), (float)(randset[7] % 100));
            get_rand((unsigned int *)randset);
            __m256 rand_val32 = _mm256_set_ps((float)(randset[0] % 2), (float)(randset[1] % 2), (float)(randset[2] % 2), (float)(randset[3] % 2), (float)(randset[4] % 2), (float)(randset[5] % 2), (float)(randset[6] % 2), (float)(randset[7] % 2));
            __m256 cmp_val31 = _mm256_cmp_ps(rand_val31, get_thresh, _CMP_GT_OQ);
            __m256 cmp_val32 = _mm256_cmp_ps(rand_val32, comp_val2, _CMP_EQ_OQ);
            get_rand((unsigned int *)randset);
            __m256 calc_pos3 = _mm256_set_ps((float)(randset[0] % ((int)val + 1)), (float)(randset[1] % ((int)val + 1)), (float)(randset[2] % ((int)val + 1)), (float)(randset[3] % ((int)val + 1)), (float)(randset[4] % ((int)val + 1)), (float)(randset[5] % ((int)val + 1)), (float)(randset[6] % ((int)val + 1)), (float)(randset[7] % ((int)val + 1)));
            get_rand((unsigned int *)randset);
            __m256 calc_neg3 = _mm256_set_ps((float)-(randset[0] % ((int)val + 1)), (float)(-randset[1] % ((int)val + 1)), (float)(-randset[2] % ((int)val + 1)), (float)(-randset[3] % ((int)val + 1)), (float)(-randset[4] % ((int)val + 1)), (float)(-randset[5] % ((int)val + 1)), (float)(-randset[6] % ((int)val + 1)), (float)(-randset[7] % ((int)val + 1)));

            __m256 blend_31 = _mm256_blendv_ps(calc_pos2, calc_neg3, cmp_val32);
            __m256 add_val31 = _mm256_add_ps(gray_channel3, blend_31);
            __m256 blend_32 = _mm256_blendv_ps(add_val31, gray_channel3, cmp_val31);

            __m256 bound_31 = _mm256_min_ps(blend_32, comp_val1);
            __m256 bound_32 = _mm256_max_ps(bound_31, comp_val2);

            _mm256_store_ps(gray_pixel + i + 16, bound_32);





            __m256 gray_channel4 = _mm256_load_ps(gray_pixel + i + 24);
            get_rand((unsigned int *)randset);
            __m256 rand_val41 = _mm256_set_ps((float)(randset[0] % 100), (float)(randset[1] % 100), (float)(randset[2] % 100), (float)(randset[3] % 100), (float)(randset[4] % 100), (float)(randset[5] % 100), (float)(randset[6] % 100), (float)(randset[7] % 100));
            get_rand((unsigned int *)randset);
            __m256 rand_val42 = _mm256_set_ps((float)(randset[0] % 2), (float)(randset[1] % 2), (float)(randset[2] % 2), (float)(randset[3] % 2), (float)(randset[4] % 2), (float)(randset[5] % 2), (float)(randset[6] % 2), (float)(randset[7] % 2));
            __m256 cmp_val41 = _mm256_cmp_ps(rand_val41, get_thresh, _CMP_GT_OQ);
            __m256 cmp_val42 = _mm256_cmp_ps(rand_val42, comp_val2, _CMP_EQ_OQ);
            get_rand((unsigned int *)randset);
            __m256 calc_pos4 = _mm256_set_ps((float)(randset[0] % ((int)val + 1)), (float)(randset[1] % ((int)val + 1)), (float)(randset[2] % ((int)val + 1)), (float)(randset[3] % ((int)val + 1)), (float)(randset[4] % ((int)val + 1)), (float)(randset[5] % ((int)val + 1)), (float)(randset[6] % ((int)val + 1)), (float)(randset[7] % ((int)val + 1)));
            get_rand((unsigned int *)randset);
            __m256 calc_neg4 = _mm256_set_ps((float)-(randset[0] % ((int)val + 1)), (float)(-randset[1] % ((int)val + 1)), (float)(-randset[2] % ((int)val + 1)), (float)(-randset[3] % ((int)val + 1)), (float)(-randset[4] % ((int)val + 1)), (float)(-randset[5] % ((int)val + 1)), (float)(-randset[6] % ((int)val + 1)), (float)(-randset[7] % ((int)val + 1)));

            __m256 blend_41 = _mm256_blendv_ps(calc_pos4, calc_neg4, cmp_val42);
            __m256 add_val41 = _mm256_add_ps(gray_channel4, blend_41);
            __m256 blend_42 = _mm256_blendv_ps(add_val41, gray_channel4, cmp_val41);

            __m256 bound_41 = _mm256_min_ps(blend_42, comp_val1);
            __m256 bound_42 = _mm256_max_ps(bound_41, comp_val2);

            _mm256_store_ps(gray_pixel + i + 24, bound_42);



			__m256 gray_channel5 = _mm256_load_ps(gray_pixel + i + 32);
            get_rand((unsigned int *)randset);
            __m256 rand_val51 = _mm256_set_ps((float)(randset[0] % 100), (float)(randset[1] % 100), (float)(randset[2] % 100), (float)(randset[3] % 100), (float)(randset[4] % 100), (float)(randset[5] % 100), (float)(randset[6] % 100), (float)(randset[7] % 100));
            get_rand((unsigned int *)randset);
            __m256 rand_val52 = _mm256_set_ps((float)(randset[0] % 2), (float)(randset[1] % 2), (float)(randset[2] % 2), (float)(randset[3] % 2), (float)(randset[4] % 2), (float)(randset[5] % 2), (float)(randset[6] % 2), (float)(randset[7] % 2));
            __m256 cmp_val51 = _mm256_cmp_ps(rand_val51, get_thresh, _CMP_GT_OQ);
            __m256 cmp_val52 = _mm256_cmp_ps(rand_val52, comp_val2, _CMP_EQ_OQ);
            get_rand((unsigned int *)randset);
            __m256 calc_pos5 = _mm256_set_ps((float)(randset[0] % ((int)val + 1)), (float)(randset[1] % ((int)val + 1)), (float)(randset[2] % ((int)val + 1)), (float)(randset[3] % ((int)val + 1)), (float)(randset[4] % ((int)val + 1)), (float)(randset[5] % ((int)val + 1)), (float)(randset[6] % ((int)val + 1)), (float)(randset[7] % ((int)val + 1)));
            get_rand((unsigned int *)randset);
            __m256 calc_neg5 = _mm256_set_ps((float)-(randset[0] % ((int)val + 1)), (float)(-randset[1] % ((int)val + 1)), (float)(-randset[2] % ((int)val + 1)), (float)(-randset[3] % ((int)val + 1)), (float)(-randset[4] % ((int)val + 1)), (float)(-randset[5] % ((int)val + 1)), (float)(-randset[6] % ((int)val + 1)), (float)(-randset[7] % ((int)val + 1)));

            __m256 blend_51 = _mm256_blendv_ps(calc_pos5, calc_neg5, cmp_val52);
            __m256 add_val51 = _mm256_add_ps(gray_channel5, blend_51);
            __m256 blend_52 = _mm256_blendv_ps(add_val51, gray_channel5, cmp_val51);

            __m256 bound_51 = _mm256_min_ps(blend_52, comp_val1);
            __m256 bound_52 = _mm256_max_ps(bound_51, comp_val2);

            _mm256_store_ps(gray_pixel + i + 32, bound_52);
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
	if (!TEST_MODE)
		times = atoi(argv[1]);
	Mat img = imread("image2.jpg");
	tv_60(img, times);

	cout << "It takes "
		 << dsum * microseconds::period::num / microseconds::period::den * 1000.0 / double(times)
		 << " milliseconds" << endl;
	return 0;
}