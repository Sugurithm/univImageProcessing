#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define FILE_NAME "image2.png"
#define MIN_HSVCOLOR Scalar(5, 30, 60)
#define MAX_HSVCOLOR Scalar(20, 150, 255)

Mat createMask(const Mat& frame) {
    Mat hsvFrame, mask; // blurred は使わないので宣言を削除
    cvtColor(frame, hsvFrame, COLOR_BGR2HSV); // BGR→HSV変換
    inRange(hsvFrame, MIN_HSVCOLOR, MAX_HSVCOLOR, mask); // 2-A 色抽出
    return mask;
}

Mat createBlurMask(const Mat& frame) {
    Mat hsvFrame, mask, blurred_hsv; // blurred_hsv に名前を変更して分かりやすく
    cvtColor(frame, hsvFrame, COLOR_BGR2HSV); // BGR→HSV変換
    // GaussianBlur の結果を blurred_hsv に格納
    GaussianBlur(hsvFrame, blurred_hsv, Size(21, 21), 20); // 平滑化
    // ぼかした画像 (blurred_hsv) を使って色抽出を行う
    inRange(blurred_hsv, MIN_HSVCOLOR, MAX_HSVCOLOR, mask); // 2-A 色抽出
    return mask;
}

Mat createMorphMask(const Mat& frame) {
    Mat hsvFrame, mask, blurred_hsv; // blurred_hsv に名前を変更して分かりやすく
    cvtColor(frame, hsvFrame, COLOR_BGR2HSV); // BGR→HSV変換
    GaussianBlur(hsvFrame, blurred_hsv, Size(21, 21), 20); // 平滑化
    inRange(blurred_hsv, MIN_HSVCOLOR, MAX_HSVCOLOR, mask); // 2-A 色抽出

    // モロフォジー処理
    morphologyEx(mask, mask, MORPH_OPEN, Mat(), Point(-1, -1), 3); // オープニング
    morphologyEx(mask, mask, MORPH_CLOSE, Mat(), Point(-1, -1), 3); // クローシング
    return mask;
}


int main() {
    // 画像を読み込む
    Mat rawFrame = imread(FILE_NAME);
    if (rawFrame.empty()) {
        cerr << "error: image cannot be read" << endl;
        return -1;
    }

    Mat originalMask = createMask(rawFrame);
    Mat blurredMask = createBlurMask(rawFrame);
    Mat morphMask = createMorphMask(rawFrame);
    Mat blurredOriginal;
    GaussianBlur(rawFrame, blurredOriginal, Size(21, 21), 20);

    Mat originalMask_colored, blurredMask_colored, morphMask_colored;
    cvtColor(originalMask, originalMask_colored, COLOR_GRAY2BGR); // 1ch -> 3ch
    cvtColor(blurredMask, blurredMask_colored, COLOR_GRAY2BGR); // 1ch -> 3ch
    cvtColor(morphMask, morphMask_colored, COLOR_GRAY2BGR); // 1ch -> 3ch

    // 1. 上段 (Original と Blurred Original と、埋め合わせ用の空の画像) を水平に結合
    Mat emptyPlaceholder = Mat::zeros(rawFrame.size(), rawFrame.type());
    Mat topRow;
    hconcat(vector<Mat>{rawFrame, blurredOriginal, emptyPlaceholder}, topRow);

    // 2. 下段 (Original Mask, Blurred Mask, Morphological Mask) を水平に結合
    Mat bottomRow;
    hconcat(vector<Mat>{originalMask_colored, blurredMask_colored, morphMask_colored}, bottomRow);

    // 3. 上段と下段を垂直に結合して最終画像を作成
    Mat combinedImage;
    vconcat(vector<Mat>{topRow, bottomRow}, combinedImage);
    
    // ウィンドウタイトルも分かりやすく変更
    imshow("Combined Images (2x3 Grid)", combinedImage);
    waitKey(0); // 何かキーが押されるまで表示を維持

    return 0;
}