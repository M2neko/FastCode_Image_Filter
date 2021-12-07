Team 04

Authors: Bingwei Wang & Grace Sun & Rui Wang

Andrew ID: bingweiw & taiges & rw3

Files in zip:
Readme.txt, CMakeList.txt, Images 1-5, Source code.

Source Code:
brightness.cpp brightness_ref.cpp duo_tone.cpp duo_tone_ref.cpp 60s_TV.cpp 60s_TV_ref.cpp avx_mathfun.h(referenced)

Running Instructions:
1. Use `cmake CMakeLists.txt` to generate Makefile, and run `make` to generate the execuatable files.
2. Run `./{EXECUATABLE_FILE_NAME} {RUNNING_TIMES}` to run our code. Example: `./brightness 100` (run brightness 100 times).
3. REF files are the baseline codes for comparasion.
4. Toggle the bar to change values in duo_tone and 60s_TV.
5. Simply enter `q` to quit the program.
6. Change image file name in each file `Mat img = imread("image2.jpg");` to use the different input image.
7. Change TEST_MODE in each file to make it running in test mode or not. TEST_MODE = 0: the program will run RUNNING_TIMES times to show the performance(average running time). TEST_MODE = 1: the program will show the image with GUI to run the image filtering.

Notes:
1. The project is using opencv2, avx, avx2, and openmp. Please make sure you have these installed in your machine.
2. For different machines, you may change the linking in CMakeLists.txt (Line 30) to modify the flag.
3. The current linking is for Mac, "-I/usr/local/include -L/usr/local/lib". If you are using Windows or Ubuntu, you may delete these flags as well as "-lomp".
4. You may use `cmake -DCMAKE_C_COMPILER="/usr/local/opt/llvm/bin/clang" -DCMAKE_CXX_COMPILER="/usr/local/opt/llvm/bin/clang++" CMakeLists.txt` (Mac) to compile.
5. "Brightness" is different from the other two files, you need to enter the value in the stdin. (Also, ou should enter the value first to begin the program).
