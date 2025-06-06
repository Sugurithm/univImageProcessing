#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define MIN_HSVCOLOR Scalar(0, 50, 60)      // 肌色の範囲
#define MAX_HSVCOLOR Scalar(20, 150, 255)   // 肌色の範囲
#define MIN_AREA_RATIO 0.03                 // 面積の最小値の割合
#define MAX_AREA_RATIO 0.3                  // 面積の最大値の割合
#define EPSILON 0.01                        // 輪郭の近似精度

void playVideo(const string& filename);                                 // 動画の再生
void sortEdgeInfo(vector<pair<int, vector<Point>>>& edgeInfo);          // 辺の数でソート
Point calculateContourCenter(const vector<Point>& contour);             // 輪郭の重心を計算
Mat createMask(const Mat& frame);                                       // マスクの作成
Mat filterMaskByArea(const Mat& inputMask, int minArea, int maxArea);   // 面積によるマスクのフィルタリング
Mat visualizeEdgeCount(const Mat& mask);                                // 辺の数を可視化
Mat filterMaskByEdges(const Mat& inputMask);                            // 辺の数によるマスクのフィルタリング
Mat changeHandColor(const Mat& currentFrame, const Mat& prevFrame, const Mat& handMask, size_t totalPixels); // 手の色を変更

Mat createMask(const Mat& frame) {
    Mat hsvFrame, mask, blurredHsv;
    cvtColor(frame, hsvFrame, COLOR_BGR2HSV);                       // BGR→HSV変換
    GaussianBlur(hsvFrame, blurredHsv, Size(5, 5), 1);              // 平滑化
    inRange(hsvFrame, MIN_HSVCOLOR, MAX_HSVCOLOR, mask);            // 2-A 色抽出
    morphologyEx(mask, mask, MORPH_OPEN, Mat(), Point(-1, -1), 5);  // オープニング
    morphologyEx(mask, mask, MORPH_CLOSE, Mat(), Point(-1, -1), 5); // クローシング
    return mask;
}

Mat filterMaskByArea(const Mat& inputMask, int minArea, int maxArea) {
    vector<vector<Point>> contours;                                             // 輪郭を格納する変数
    findContours(inputMask.clone(), contours, RETR_LIST, CHAIN_APPROX_NONE);    // マスクのクローンを作成してから輪郭を見つける．元のマスクを変更しないため．
    Mat filteredMask = Mat::zeros(inputMask.size(), CV_8UC1);                   // 輪郭を描画するためのマスク（初期設定は黒）

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]); // 各領域の面積を計算
        if ( minArea <= area && area <= maxArea ) {
            // 面積条件を満たす輪郭を白色で描画
            drawContours(filteredMask, contours, i, Scalar(255), FILLED);
        }
    }

    return filteredMask;
}

// 輪郭の重心を計算
// これは，edge処理で情報を表示する箇所を決めるため実装
Point calculateContourCenter(const vector<Point>& contour) {
    Moments m = moments(contour); // 空間モーメントを求め，格納
    if (m.m00 > 0) {
        return Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
    } else {
        return Point(-1, -1);
    }
}

// 辺の数を可視化
Mat visualizeEdgeCount(const Mat& mask) {

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
        // 辺の数を表示する
        int edgeCount = approxContours[i].size();
        Point center = calculateContourCenter(approxContours[i]);
        circle(contourImage, center, 5, Scalar(0, 255, 255), 2); // 重心を描画
        putText(contourImage, to_string(edgeCount), center, FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2); // 辺の数を描画
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

// 動画再生
void playVideo(const string& filename) {
    VideoCapture video(filename);
    if (!video.isOpened()) return; // 開けなかった場合

    while (true) {
        Mat frame;
        video >> frame;
        if (frame.empty()) break; // 空だった場合
        imshow("Recorded Video", frame);
        if (waitKey(1) == 'q') {
            break;
        }
    }
    destroyWindow("Recorded Video");
}

Mat changeHandColor(const Mat& currentFrame, const Mat& prevFrame, const Mat& handMask, size_t totalPixels) {
    
    Mat diffFrame, grayCurrent, grayPrev;
    cvtColor(currentFrame, grayCurrent, COLOR_BGR2GRAY);    // グレースケール変換
    cvtColor(prevFrame, grayPrev, COLOR_BGR2GRAY);          // グレースケール変換

    absdiff(grayCurrent, grayPrev, diffFrame);                  // フレーム間差分
    threshold(diffFrame, diffFrame, 100, 255, THRESH_BINARY);   // 閾値処理

    // 面積計算
    vector<vector<Point>> contours;
    findContours(diffFrame, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    double movingArea = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        movingArea += area;
    }

    Mat resultFrame = currentFrame.clone(); // 手の領域だけを取り出す

    // 動いていれば赤色，動いていなければ元の色
    if (movingArea > totalPixels * 0.0001) {
        resultFrame.setTo(Scalar(0, 0, 255), handMask);
    }

    // クロマキー編集：手以外を背景画像に差し替え
    Mat handMaskInv;
    bitwise_not(handMask, handMaskInv); // 手領域以外
    Mat bgResized, wallpaperImage;
    wallpaperImage = imread("wallpaperImage.jpg");
    resize(wallpaperImage, bgResized, currentFrame.size()); // 背景画像をリサイズ

    bgResized.copyTo(resultFrame, handMaskInv); // 手以外に背景画像を貼る

    return resultFrame;
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
    flip(rawFrame, rawFrame, 1);
    Mat prevFrame = rawFrame.clone();

    // ========================= 動画保存 ==========================

    VideoWriter video;                  // 動画保存用オブジェクト
    bool isRecording = false;           // 録画状態を管理するフラグ
    string videoFilename = "movie.avi"; // 保存する動画ファイル名

    int ssCount = 0;

    // ========================= 定数宣言 ==========================

    size_t totalPixels = rawFrame.rows * rawFrame.cols;     // 全画素数
    cout << "Total Pixels: " << totalPixels << endl;
    int calculateMinArea = totalPixels * MIN_AREA_RATIO;    // 最小面積
    int calculateMaxArea = totalPixels * MAX_AREA_RATIO;    // 最大面積
    cout << "Min Area: " << calculateMinArea << endl;
    cout << "Max Area: " << calculateMaxArea << endl;
    int imageCount = 0;                                     // 画像数（保存用）

    // ========================== メインループ ==========================

    while (true) {
        camera >> rawFrame;
        if (rawFrame.empty()) break;
        flip(rawFrame, rawFrame, 1); // 左右反転

        // ==================== マスク処理 ====================

        Mat mask = createMask(rawFrame); // マスク処理
        Mat filteredMask = filterMaskByArea(mask, calculateMinArea, calculateMaxArea);  // 面積によるフィルタリング
        Mat filteredMask2 = visualizeEdgeCount(filteredMask);                           // 辺の数を可視化
        Mat filteredMask3 = filterMaskByEdges(filteredMask);                            // 辺の数によるフィルタリング

        // ==================== 色変化処理 ====================
        
        Mat coloredHandFrame = changeHandColor(rawFrame, prevFrame, filteredMask3, totalPixels);
        prevFrame = rawFrame.clone();

        // ==================== 表示 ====================

        Mat maskBGR, filteredMaskBGR;
        cvtColor(mask, maskBGR, COLOR_GRAY2BGR);                    // 1ch -> 3ch
        cvtColor(filteredMask, filteredMaskBGR, COLOR_GRAY2BGR);    // 1ch -> 3ch
        cvtColor(filteredMask3, filteredMask3, COLOR_GRAY2BGR);     // 1ch -> 3ch

        // divider
        int dividerWidth = 3;
        Mat divider = Mat(rawFrame.rows, dividerWidth, CV_8UC3, Scalar(0, 0, 255));

        // 結合
        Mat combinedImage, row1, row2;
        hconcat(vector<Mat>{rawFrame, divider, maskBGR, divider, filteredMaskBGR}, row1);
        hconcat(vector<Mat>{filteredMask2, divider, filteredMask3, divider, coloredHandFrame}, row2);
        vconcat(vector<Mat>{row1, row2}, combinedImage);
        imshow("View Edge | Filtered Mask by Edges", combinedImage);

        // =================== 動画保存 ====================

        // ===== 保存する映像作成 =====

        Mat recordCombinedImage;
        /*
        original | createMask | filteredMaskByArea  |
        visualizeEdgeCount | filteredMaskByEdges | changeHandColor
        水平に合成した2枚の画像を垂直に繋ぐ
        */
        hconcat(vector<Mat>{rawFrame, divider, maskBGR, divider, filteredMaskBGR}, row1);
        hconcat(vector<Mat>{filteredMask2, divider, filteredMask3, divider, coloredHandFrame}, row2);
        vconcat(vector<Mat>{row1, row2}, recordCombinedImage);
        video << recordCombinedImage;

        if (isRecording) {
            // 録画中ならば
            if (!video.isOpened()) {
                // 動画が開かれていなければ，動画ファイルを開く
                video.open(videoFilename, VideoWriter::fourcc('M','J','P','G'), 30, recordCombinedImage.size(), true);
                if (!video.isOpened()) {
                    isRecording = false;
                    cerr << "Failed to open video file" << endl;
                }
            }
            if (video.isOpened()) {
                // 動画保存
                video << recordCombinedImage;
            }
        } else {
            // 録画中でなければ
            if (video.isOpened()) {
                video.release(); // 動画を閉じる
            }
        }

        // ==================== キー入力 ====================

        int key = waitKey(1);
        if (key == 'q') {
            break;
        } else if (key == 's') {    // 画像保存
            cerr << "pushed 's' " << imageCount << endl;
            string filename = "image" + to_string(imageCount) + ".png";
            imwrite(filename, coloredHandFrame);
            imageCount++;           // 画像の数を増やす
        } else if (key == 'r') {    // 録画の切り替え
            cerr << "pushed 'r' " << isRecording << endl;
            isRecording = !isRecording;
        } else if (key == 'p') {    // 録画中の動画再生
            cerr << "pushed 'p' " << video.isOpened() << endl;
            if (!video.isOpened()) playVideo(videoFilename);
        }
    }

    // ==================== 例外処理 ====================

    // ループから抜ける時，録画中であれば録画を終了する必要がある．
    if (video.isOpened()) {
        video.release();
    }

    camera.release();       // カメラリソースを解放
    destroyAllWindows();    // すべてのウィンドウを閉じる

    return 0;
}