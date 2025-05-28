// exp1.cpp 色を抽出する
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// #define MIN_HSVCOLOR Scalar(0, 60, 80)
// #define MAX_HSVCOLOR Scalar(10, 160, 240)

#define MIN_HSVCOLOR Scalar(0, 30, 60)
#define MAX_HSVCOLOR Scalar(20, 150, 255)
#define MIN_AREA 0
#define MAX_AREA 10000000

Mat createMask(const Mat& frame) {
    Mat blurred, hsvImage, mask;
    GaussianBlur(frame, blurred, Size(3, 3), 1); // 平滑化
    cvtColor(blurred, hsvImage, COLOR_BGR2HSV); // HSV変換
    inRange(hsvImage, MIN_HSVCOLOR, MAX_HSVCOLOR, mask); // 色によるマスク
    erode(mask, mask, Mat(), Point(-1, -1), 3); // 縮小
    dilate(mask, mask, Mat(), Point(-1, -1), 3); // 膨張
    return mask;
}

Mat filterMaskByArea(const Mat& inputMask) {
    vector<vector<Point>> contours;
    // マスクのクローンを作成してから輪郭を見つける．元のマスクを変更しないため．
    findContours(inputMask.clone(), contours, RETR_LIST, CHAIN_APPROX_NONE);

    Mat filteredMask = Mat::zeros(inputMask.size(), CV_8UC1); // 輪郭を描画するためのマスク（初期設定は黒）

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if ( MIN_AREA <= area && area <= MAX_AREA) {
            // 面積条件を満たす輪郭を白色で描画
            drawContours(filteredMask, contours, i, Scalar(255), FILLED);
        }
    }
    return filteredMask;
}


int main() {
    VideoCapture camera(1);
    if (!camera.isOpened()) {
        cerr << "Camera Open Failed" << endl;
        return -1;
    }

    int imageCount = 0;
    while (true) {
        Mat rawFrame; 
        camera >> rawFrame; // 1-A カメラから1フレーム読み込む
        if (rawFrame.empty()) break;

        flip(rawFrame, rawFrame, 1); // 左右反転

        // ===== 処理 =====
        Mat mask = createMask(rawFrame); // マスク処理

        Mat filteredMask = filterMaskByArea(mask); // 面積によるフィルタリング

        // ===== 表示 =====
        Mat maskBGR, filteredMaskBGR;
        cvtColor(mask, maskBGR, COLOR_GRAY2BGR); // 1ch -> 3ch
        cvtColor(filteredMask, filteredMaskBGR, COLOR_GRAY2BGR); // 1ch -> 3ch

        Mat combinedImage;
        hconcat(vector<Mat>{rawFrame, maskBGR, filteredMaskBGR}, combinedImage);
        imshow("Original | Color Mask | Filtered Mask by Area", combinedImage);


        // ===== キー入力 ====
        int key = waitKey(1);
        if (key == 'q') break;
        else if (key == 's') {
            cout << "Saved Image" << endl;
            imwrite("image" + to_string(imageCount) + ".png", rawFrame);
            imageCount++;
        }
    }

    camera.release(); // カメラリソースを解放
    destroyAllWindows(); // すべてのウィンドウを閉じる

    return 0;
}