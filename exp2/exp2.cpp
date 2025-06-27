#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;

#define FILEPATH "sample3.png"
#define MARGIN 50

// ==================== グローバル変数 ====================

vector<float> g_dominantAngles; // 奥行き方向の角度のリスト
double g_degree; // 奥行き方向の角度

// マウスコールバック関数とメインループ間で共有される
bool g_roi_selecting = false; // ROI選択中フラグ
int g_click_count = 0; // クリック回数¥
Rect g_roi_rect = Rect(-1, -1, 0, 0); // 選択されたROIの矩形
Point g_start_point = Point(-1, -1); // ROI選択開始点
Point g_current_mouse_pos = Point(-1, -1); // マウスの現在の位置

// select_roi関数内で共有される
Mat g_initial_frame;

// init_output_file関数内で共有される
Mat g_template_image; // テンプレート画像
Point g_previous_center; // 前のフレームの重心位置
bool g_is_first_frame = true; // 最初のフレームかどうかを判断するフラグ

// process_tracking関数内で共有される
Rect g_last_tracked_rect; // 前回の追跡位置

ofstream g_output_file;


Mat computeGradient(const Mat& frame) {
    Mat grayImage;
    cvtColor(frame, grayImage, COLOR_BGR2GRAY); // グレースケールに変換

    // Sobelフィルタのカーネル
    Mat sobel_x = (Mat_<double>(3, 3) << -3, 0, 3, -5, 0, 5, -3, 0, 3);
    Mat sobel_y = (Mat_<double>(3, 3) << -3, -5, -3, 0, 0, 0, 3, 5, 3);

    // オペレータを適用
    Mat grad_x, grad_y;
    filter2D(grayImage, grad_x, -1, sobel_x);
    filter2D(grayImage, grad_y, -1, sobel_y);

    // 画素値の絶対値をとる．
    Mat abs_grad_x, abs_grad_y, grad_combined;
    convertScaleAbs(grad_x, abs_grad_x); // 画素値の絶対値をとる
    convertScaleAbs(grad_y, abs_grad_y); // 画素値の絶対値をとる
    add(abs_grad_x, abs_grad_y, grad_combined); // 勾配強度を結合

    return grad_combined;
}

Mat HoughTransform(const Mat& gradImage, const Mat& frame, vector<float>& dominantAngles) {

    // 勾配画像の2値化（THRESH_OTSU：クラス間分散を最大にするしきい値を自動計算）
    Mat edges;
    threshold(gradImage, edges, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // HoughLinesPを用いてエッジ画像から直線を検出
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 30, 200, 5);

    // === 検出した直線を元画像に描画して保存 ===
    Mat line_overlay = frame.clone();
    for (size_t i = 0; i < lines.size(); ++i) {
        Vec4i l = lines[i];
        line(line_overlay, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
    }
    imwrite("line.png", line_overlay);  // ファイル保存

    // 検出された直線の角度を格納するMat
    Mat angles(static_cast<int>(lines.size()), 1, CV_32F);

    // 各直線の傾きを計算し，格納
    for (size_t i = 0; i < lines.size(); ++i) {
        float dx = lines[i][2] - lines[i][0]; // x座標の差
        float dy = lines[i][3] - lines[i][1]; // y座標の差
        angles.at<float>(i, 0) = atan2(dy, dx); // 角度を計算し，格納
    }

    if (lines.empty()) return frame.clone(); // 直線を検出しなかった場合は元の画像を返す

    // k-meansによるクラスタリング
    const int K = 5; // クラスタ数
    Mat labels, centers;

    kmeans(angles, K, labels, 
        TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0.001), 
        10, KMEANS_PP_CENTERS, centers);

    // 各クラスタに属する要素数をカウント
    vector<int> clusterCounts(K, 0);
    for (int i = 0; i < labels.rows; ++i) clusterCounts[labels.at<int>(i, 0)]++;

    // 要素数・クラスタのペアを作成
    vector<pair<int, int>> sortedClusters;
    for (int i = 0; i < K; ++i) sortedClusters.push_back({clusterCounts[i], i});
    
    // 要素数が大きい順にソート
    sort(sortedClusters.rbegin(), sortedClusters.rend());

    // 上位2つのクラスタIDを取得
    vector<int> top2ClusterIDs;
    if (K >= 1 && sortedClusters[0].first > 0) top2ClusterIDs.push_back(sortedClusters[0].second);
    if (K >= 2 && sortedClusters[1].first > 0) top2ClusterIDs.push_back(sortedClusters[1].second);

    // 上位クラスタの中心角度を格納
    for (int cluster_id : top2ClusterIDs) {
        float angle_rad = centers.at<float>(cluster_id, 0);
        dominantAngles.push_back(angle_rad);
    }

    // 出力画像を返す（可視化は別処理で行う）
    return frame.clone();
}

// マウスイベントのコールバック関数
void on_mouse(int event, int x, int y, int, void*) {

    if (event == EVENT_LBUTTONDOWN) { // 左クリックされた場合

        if (!g_roi_selecting) { // ROI選択中でない場合
            g_start_point = Point(x, y);    // 開始座標を記録
            g_roi_selecting = true;         // ROI選択中にする
            g_click_count = 1;              // クリック数を1にする

        } else { // ROI選択中の場合
            Point end_point(x, y);          // 終了座標を記録
            g_roi_rect = Rect(              // 矩形を作成（正規化）
                min(g_start_point.x, end_point.x),
                min(g_start_point.y, end_point.y), 
                abs(g_start_point.x - end_point.x), 
                abs(g_start_point.y - end_point.y)
            );
            g_roi_selecting = false;        // ROI選択中を解除
            g_click_count = 2;              // クリック数を2にする
            g_current_mouse_pos = Point(-1, -1);    // 現在のマウス座標を-1,-1にする
        }

    } else if (event == EVENT_MOUSEMOVE && g_roi_selecting) { // マウス移動中でROI選択中の場合
        g_current_mouse_pos = Point(x, y); // 現在のマウス座標を更新
    }
}

// 洗濯中のUIを整える
bool select_roi(VideoCapture& cap, const string& window_name) {
    cap >> g_initial_frame;
    if (g_initial_frame.empty()) return false; 

    // ROI選択を行う
    while (g_click_count < 2) {
        Mat display = g_initial_frame.clone();

        // ROI選択中の場合は矩形を描画
        if (g_roi_selecting && g_current_mouse_pos.x != -1)
            rectangle(display, g_start_point, g_current_mouse_pos, Scalar(0, 255, 0), 2);

        imshow(window_name, display); // 表示

        if (waitKey(1) == 'q') return false;
    }

    return true;
}

// 射影変換を行う
double projectedLength(double dx, double dy, double degree) {
    double radian = degree * (M_PI / 180);
    double ux = cos(radian);
    double uy = sin(radian);
    double distance = abs(dx * ux + dy * uy);
    return distance;
}

// 初期テンプレート画像設定
bool initialize_template() {
    if (g_roi_rect.width <= 0 || g_roi_rect.height <= 0) return false;

    g_template_image = g_initial_frame(g_roi_rect).clone(); // ROIからテンプレート画像を抽出
    g_previous_center = Point(
        g_roi_rect.x + g_roi_rect.width / 2, 
        g_roi_rect.y + g_roi_rect.height / 2
    );
    g_is_first_frame = false;
    imshow("Template", g_template_image);

    return true;
}

bool process_tracking_frame(Mat& frame, const string& window_name) {
    Mat display_frame = frame.clone();
    if (g_template_image.empty()) return false;

    // ==================== 探索ROIの設定 ====================
    // 検出ROIを設定
    Rect search_roi;
    if (g_last_tracked_rect.width > 0 || g_last_tracked_rect.height > 0) {
        // 前フレームで探索が正しく行われた場合，前フレームの探索ROIを使用
        search_roi = g_last_tracked_rect;
    } else {
        // 前フレームで探索が正しく行われなかった場合，画面全体を探索ROIとする
        search_roi = Rect(0, 0, frame.cols, frame.rows);
    }
    search_roi -= Point(MARGIN, MARGIN); // ポインタを左上に移す（領域が広くなるため）
    search_roi += Size(MARGIN * 2, MARGIN * 2); // マージン分領域を広くする
    search_roi &= Rect(0, 0, frame.cols, frame.rows); // 画面内に収まるように調整
    
    // ==================== スケーリング & マッチング ====================
    vector<float> scale_factors = {1.0f, 0.95f, 1.05f}; // スケールファクタを設定
    double best_score = -1.0; // 最高スコアを-1.0で初期化
    Point best_matchLoc; // 最高スコアの"座標"
    Rect best_tracked_rect; // 最高スコアの"矩形"
    Mat best_template; // 最高スコアの"テンプレート"

    for (float scale : scale_factors) {
        // スケーリングをする．
        Mat scaled_template;
        resize(g_template_image, scaled_template, Size(), scale, scale);
        if (scaled_template.cols > search_roi.width || scaled_template.rows > search_roi.height) continue;

        Mat result;
        matchTemplate(frame(search_roi), scaled_template, result, TM_CCOEFF_NORMED); // マッチング

        double maxVal; Point maxLoc;
        minMaxLoc(result, 0, &maxVal, 0, &maxLoc); // 最高スコアを更新
        if (maxVal > best_score) {
            best_score = maxVal;
            best_matchLoc = maxLoc + search_roi.tl();
            best_tracked_rect = Rect(best_matchLoc.x, best_matchLoc.y, scaled_template.cols, scaled_template.rows);
        }
    }
    if (best_score < 0.5) return false; // 最高スコアが0.5以下の場合，検出失敗

    // ==================== 移動量 ====================
    Point current_center( // 現在の中心座標
        best_tracked_rect.x + best_tracked_rect.width / 2, 
        best_tracked_rect.y + best_tracked_rect.height / 2
    );
    static int frame_count = 0; // フレーム数
    frame_count++;
    // 直線の単位ベクトルを計算
    int dx = current_center.x - g_previous_center.x; // 前フレームからの移動量
    int dy = current_center.y - g_previous_center.y; // 前フレームからの移動量
    // 変位ベクトルの直線への写像は，直線の単位ベクトルとの内積と等価．
    double distance = projectedLength(dx, dy, g_degree);

    // ==================== CSV出力 ====================
    if (g_output_file.is_open() && (dx != 0 || dy != 0)) {
        g_output_file   << frame_count << "," 
                        << current_center.x << "," 
                        << current_center.y << ","
                        << dx << "," 
                        << dy << "," 
                        << distance << "," 
                        << best_score << endl;
    }

    g_previous_center = current_center;

    // ==================== 矩形の描画 ====================
    rectangle(display_frame, best_tracked_rect, Scalar(0, 0, 255), 2);
    circle(display_frame, current_center, 5, Scalar(0, 255, 0), -1);
    putText(display_frame, "Center: (" + to_string(current_center.x) + ", " + to_string(current_center.y) + ")",
            Point(current_center.x + 10, current_center.y + 10), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
    putText(display_frame, "Score: " + to_string(best_score),
            Point(current_center.x + 10, current_center.y + 50), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
    if (dx != 0 || dy != 0 ) {
        putText(display_frame, "dx: " + to_string(dx) + ", dy: " + to_string(dy),
                Point(current_center.x + 10, current_center.y + 90), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
        putText(display_frame, "Distance: " + to_string(distance),
                Point(current_center.x + 10, current_center.y + 130), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
    }
    // ==================== 直線の描画 ====================
    Point center(display_frame.cols / 2, display_frame.rows / 2); // 画面の中心座標
    float len = max(display_frame.cols, display_frame.rows) / 2; // 直線の長さ
    for (size_t i = 0; i < g_dominantAngles.size(); i++) { // 各直線の描画
        float angle = g_dominantAngles[i];
        Point p1 = center + Point(len * cos(angle), len * sin(angle));
        Point p2 = center - Point(len * cos(angle), len * sin(angle));
        if (i == 0) {
            line(display_frame, p1, p2, Scalar(0, 255, 0), 2);
        }
        else {
            line(display_frame, p1, p2, Scalar(0, 50, 0), 2);
        }
    }

    // ==================== テンプレート更新 ====================
    g_template_image = frame(best_tracked_rect).clone(); // ROIからテンプレート画像を抽出
    g_last_tracked_rect = best_tracked_rect;

    // ==================== 表示 ====================
    imshow("Template", g_template_image);
    imshow(window_name, display_frame);

    return waitKey(1) != 'q';
}


int main() {
    // =============== 1.　入力動画の読み込み ===============
    string video_path = "traffic.mov";
    VideoCapture cap(video_path);
    if (!cap.isOpened()) return -1;

    Mat first_frame;
    cap >> first_frame;
    if (first_frame.empty()) return -1;

    // =============== 2.　勾配画像の計算 ===============
    Mat grad = computeGradient(first_frame);

    // =============== 3.　Hough変換とK-meansによる直線検出 ==============
    HoughTransform(grad, first_frame, g_dominantAngles);
    if (!g_dominantAngles.empty()) {
        g_degree = g_dominantAngles[0] * 180.0 / CV_PI; // 奥行き方向の角度を計算（ラジアンから度に変換）
    }

    // =============== 4. トラッキング ===============
    string window_name = "Camera Feed";
    namedWindow(window_name);
    setMouseCallback(window_name, on_mouse);

    g_output_file.open("tracking_data.csv");
    g_output_file << "Frame,CenterX,CenterY,dx,dy,Distance,Score\n";

    if (!select_roi(cap, window_name) || !initialize_template()) return -1;

    Mat frame;
    while (cap.read(frame)) {
        if (!process_tracking_frame(frame, window_name)) break;
    }

    cap.release();
    destroyAllWindows();
    g_output_file.close();
    return 0;
}
