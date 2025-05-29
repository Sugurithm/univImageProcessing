#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define MIN_HSVCOLOR Scalar(0, 50, 60)
#define MAX_HSVCOLOR Scalar(20, 150, 255)
#define MIN_AREA_RATIO 0.03
#define MAX_AREA_RATIO 0.3

Mat createMask(const Mat& frame) {
    Mat hsvFrame, mask, blurredHsv;
    cvtColor(frame, hsvFrame, COLOR_BGR2HSV); // BGR→HSV変換
    GaussianBlur(hsvFrame, blurredHsv, Size(5, 5), 1); // 平滑化
    inRange(hsvFrame, MIN_HSVCOLOR, MAX_HSVCOLOR, mask); // 2-A 色抽出
    morphologyEx(mask, mask, MORPH_OPEN, Mat(), Point(-1, -1), 5); // オープニング
    morphologyEx(mask, mask, MORPH_CLOSE, Mat(), Point(-1, -1), 5); // クローシング
    return mask;
}


Mat filterMaskByArea(const Mat& inputMask, int minArea, int maxArea) {
    vector<vector<Point>> contours;
    // マスクのクローンを作成してから輪郭を見つける．元のマスクを変更しないため．
    findContours(inputMask.clone(), contours, RETR_LIST, CHAIN_APPROX_NONE);

    Mat filteredMask = Mat::zeros(inputMask.size(), CV_8UC1); // 輪郭を描画するためのマスク（初期設定は黒）

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if ( minArea <= area && area <= maxArea ) {
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


    Mat rawFrame;
    camera >> rawFrame;
    if (rawFrame.empty()) return -1;
    size_t totalPixels = rawFrame.rows * rawFrame.cols;
    cout << "Total Pixels: " << totalPixels << endl;
    int calculateMinArea = totalPixels * MIN_AREA_RATIO;
    int calculateMaxArea = totalPixels * MAX_AREA_RATIO;
    cout << "Min Area: " << calculateMinArea << endl;
    cout << "Max Area: " << calculateMaxArea << endl;
    int imageCount = 0;


    while (true) {
        camera >> rawFrame; // 1-A カメラから1フレーム読み込む
        if (rawFrame.empty()) break;

        flip(rawFrame, rawFrame, 1); // 左右反転

        Mat mask = createMask(rawFrame); // マスク処理
        Mat filteredMask = filterMaskByArea(mask, calculateMinArea, calculateMaxArea); // 面積によるフィルタリング

        // ===== 表示 =====
        Mat maskBGR, filteredMaskBGR;
        cvtColor(mask, maskBGR, COLOR_GRAY2BGR); // 1ch -> 3ch
        cvtColor(filteredMask, filteredMaskBGR, COLOR_GRAY2BGR); // 1ch -> 3ch

        int dividerWidth = 3;
        Mat divider = Mat(rawFrame.rows, dividerWidth, CV_8UC3, Scalar(0, 0, 255));

        Mat combinedImage;
        hconcat(vector<Mat>{rawFrame, divider, maskBGR, divider, filteredMaskBGR}, combinedImage);
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