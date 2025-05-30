// exp1.cpp 色を抽出する
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define FILE_NAME "sample.png"
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

Point calculateContourCenter(const vector<Point>& contour) {
    Moments m = moments(contour);

    if (m.m00 > 0) {
        return Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
    } else {
        return Point(-1, -1);
    }
}

// 入力マスクから輪郭を抽出し、最大頂点数の輪郭を除外してフィルタリングされたマスクを生成する関数
Mat visualizeEdgeCount(const Mat& mask) {

    // 1. 輪郭検出
    vector<vector<Point>> contours;
    findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 2. 近似
    vector<vector<Point>> approxContours(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        approxPolyDP(contours[i], approxContours[i], 3, true);
    }

    // 3. 描画
    Mat contourImage = Mat::zeros(mask.size(), CV_8UC3);
    for (size_t i = 0; i < approxContours.size(); i++) {
        drawContours(contourImage, approxContours, (int)i, Scalar(0, 255, 0), 2); // 緑色で描画
        // 辺の数を
        int edgeCount = approxContours[i].size();
        Point center = calculateContourCenter(approxContours[i]);
        circle(contourImage, center, 5, Scalar(0, 255, 255), 2);
        putText(contourImage, to_string(edgeCount), center, FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
    }

    return contourImage;
}

int main() {
    // 画像を読み込む
    Mat rawFrame = imread(FILE_NAME);
    if (rawFrame.empty()) {
        cerr << "error: image cannot be read" << endl;
        return -1;
    }

    size_t totalPixels = rawFrame.rows * rawFrame.cols;
    cout << "Total Pixels: " << totalPixels << endl;

    Mat mask = createMask(rawFrame); // マスク処理
    Mat filteredMask = filterMaskByArea(mask, totalPixels * MIN_AREA_RATIO, totalPixels * MAX_AREA_RATIO); // 面積によるフィルタリング
    Mat filteredMask2 = visualizeEdgeCount(filteredMask); // 輪郭の近似

    // ===== 表示 =====
    Mat maskBGR, filteredMaskBGR;
    cvtColor(mask, maskBGR, COLOR_GRAY2BGR); // 1ch -> 3ch
    cvtColor(filteredMask, filteredMaskBGR, COLOR_GRAY2BGR); // 1ch -> 3ch

    int dividerWidth = 3;
    Mat divider = Mat(rawFrame.rows, dividerWidth, CV_8UC3, Scalar(0, 0, 255));

    Mat combinedImage;
    hconcat(vector<Mat>{filteredMaskBGR, divider, filteredMask2}, combinedImage);
    imshow("Filtered Mask by Area | Filtered Mask by Edges", combinedImage);

    // imshow("Filtered Mask by Edges", filteredMask2);

    waitKey(0); // 何かキーが押されるまで表示を維持

    return 0;
}