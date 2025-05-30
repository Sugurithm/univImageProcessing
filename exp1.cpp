#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define MIN_HSVCOLOR Scalar(0, 50, 60)
#define MAX_HSVCOLOR Scalar(20, 150, 255)
#define MIN_AREA_RATIO 0.03
#define MAX_AREA_RATIO 0.3
#define EPSILON 0.01

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

Mat visualizeEdgeCount(const Mat& mask) {

    // GaussianBlur(mask, mask, Size(7, 7), 1);
    // morphologyEx(mask, mask, MORPH_OPEN, Mat(), Point(-1, -1), 5);

    // 1. 輪郭検出
    vector<vector<Point>> contours;
    findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 2. 近似
    vector<vector<Point>> approxContours(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        approxPolyDP(contours[i], approxContours[i], EPSILON, true);
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

// 入力マスクから輪郭を抽出し、最大頂点数の輪郭を除外してフィルタリングされたマスクを生成する関数
void sortEdgeInfo(vector<pair<int, vector<Point>>>& edgeInfo) {
    // 降順にソート
    for (size_t i = 0; i < edgeInfo.size(); i++) {
        for (size_t j = i + 1; j < edgeInfo.size(); j++) {
            if (edgeInfo[i].first < edgeInfo[j].first) {
                swap(edgeInfo[i], edgeInfo[j]);
            }
        }
    }
}

Mat filterMaskByEdges(const Mat& inputMask) {
    // 1. 輪郭検出
    vector<vector<Point>> contours;
    findContours(inputMask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 2. 輪郭近似と辺の数を格納
    vector<pair<int, vector<Point>>> edgeInfo;
    for (size_t i = 0; i < contours.size(); i++) {
        vector<Point> approx;
        approxPolyDP(contours[i], approx, EPSILON, true);
        edgeInfo.emplace_back(approx.size(), approx);
    }

    // 3. 辺の数でソート（大きい順）
    sortEdgeInfo(edgeInfo);

    // 4. 残す輪郭の個数を決める
    int nContours = static_cast<int>(edgeInfo.size());
    int keepCount = 0;
    if (nContours == 1) {
        keepCount = 1; // 領域1つはそのまま（手だけとみなす）
    } else if (nContours == 2) {
        keepCount = 1; // 2つなら辺の多い方だけ残す
    } else if (nContours >= 3) {
        keepCount = 2; // 3つ以上なら上位2つ残す
    }

    // 5. マスク作成：指定数だけ白で描画
    Mat resultMask = Mat::zeros(inputMask.size(), CV_8UC1);
    for (int i = 0; i < keepCount; i++) {
        drawContours(resultMask, vector<vector<Point>>{edgeInfo[i].second}, -1, Scalar(255), FILLED);
    }

    // 6. 縮小処理
    erode(resultMask, resultMask, Mat(), Point(-1, -1), 1);

    return resultMask;
}

int main() {
    VideoCapture camera(0);
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

        // ===== マスク処理 =====
        Mat mask = createMask(rawFrame); // マスク処理
        Mat filteredMask = filterMaskByArea(mask, calculateMinArea, calculateMaxArea); // 面積によるフィルタリング
        Mat filteredMask2 = visualizeEdgeCount(filteredMask); // 輪郭の近似
        Mat filteredMask3 = filterMaskByEdges(filteredMask);

        // ===== 表示 =====
        Mat maskBGR, filteredMaskBGR;
        cvtColor(mask, maskBGR, COLOR_GRAY2BGR); // 1ch -> 3ch
        cvtColor(filteredMask, filteredMaskBGR, COLOR_GRAY2BGR); // 1ch -> 3ch
        cvtColor(filteredMask3, filteredMask3, COLOR_GRAY2BGR); // 1ch -> 3ch

        // 間の仕切り
        int dividerWidth = 3;
        Mat divider = Mat(rawFrame.rows, dividerWidth, CV_8UC3, Scalar(0, 0, 255));

        // 画像結合
        Mat combinedImage;
        hconcat(vector<Mat>{filteredMask2, divider, filteredMask3}, combinedImage);
        imshow("View Edge | Filtered Mask by Edges", combinedImage);

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