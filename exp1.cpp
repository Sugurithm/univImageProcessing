// exp1.cpp 色を抽出する
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define MIN_HSVCOLOR Scalar(0, 60, 80)
#define MAX_HSVCOLOR Scalar(10, 160, 240)

Mat createMask(const Mat& frame) {
    Mat hsvFrame, mask;
    cvtColor(frame, hsvFrame, COLOR_BGR2HSV); // BGR→HSV変換
    inRange(hsvFrame, MIN_HSVCOLOR, MAX_HSVCOLOR, mask); // 2-A 色抽出
    return mask;
}


int main() {
    VideoCapture camera(0);
    if (!camera.isOpened()) {
        cerr << "Camera Open Failed" << endl;
        return -1;
    }

    while (true) {
        Mat rawFrame; 
        camera >> rawFrame; // 1-A カメラから1フレーム読み込む
        if (rawFrame.empty()) break;

        flip(rawFrame, rawFrame, 1); // 左右反転

        // ===== マスク処理 =====
        Mat mask = createMask(rawFrame);

        // ===== 表示 =====

        Mat maskBGR;
        cvtColor(mask, maskBGR, COLOR_GRAY2BGR); // チャネル変換 1ch -> 3ch

        Mat combinedImage;
        hconcat(vector<Mat>{rawFrame, maskBGR}, combinedImage);
        imshow("Original | Mask", combinedImage);


        // ===== キー入力 ====
        int key = waitKey(1);
        if (key == 'q') break;
    }

    camera.release(); // カメラリソースを解放
    destroyAllWindows(); // すべてのウィンドウを閉じる

    return 0;
}