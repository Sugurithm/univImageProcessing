#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <limits>
#include <string> // std::to_string を使うために必要

using namespace cv;
using namespace std;

// ==================== グローバル変数 ====================
// マウスコールバック関数とメインループ間で共有される
Mat template_image_rgb;    // 逐次更新されるRGBテンプレート画像
Mat template_image_edge;   // 逐次更新されるテンプレート画像のエッジ画像
Mat template_image_dist;   // 逐次更新されるテンプレート画像の距離変換結果
Mat g_original_template_rgb; // 最初に選択されたオリジナルのRGBテンプレート画像 (固定表示用)

Rect g_roi_rect;           // 選択されたROIの矩形
bool g_roi_selecting = false; // ROI選択中フラグ
Point g_start_point;       // ROI選択開始点
int g_click_count = 0;     // クリック回数
Mat g_initial_frame;       // ROI選択フェーズで使用する最初のフレームのコピー
Point mouse_position = Point(-1, -1); // マウスの現在の位置

// 高速化のためのオプション（探索範囲の限定）
Rect g_last_tracked_rect; // 前フレームの追跡結果の矩形

// ==================== マウスコールバック関数 ====================
void on_mouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        if (!g_roi_selecting) { // ROI選択開始（1回目のクリック）
            g_start_point = Point(x, y);
            g_roi_selecting = true;
            g_click_count = 1;
            cout << "ROI-Start-Point: (" << x << ", " << y << ")" << endl;
        } else { // ROI選択終了（2回目のクリック）
            Point end_point(x, y);
            // 矩形の座標を正規化
            g_roi_rect = Rect(
                min(g_start_point.x, end_point.x),
                min(g_start_point.y, end_point.y),
                abs(g_start_point.x - end_point.x),
                abs(g_start_point.y - end_point.y)
            );
            g_roi_selecting = false;
            g_click_count = 2;
            cout << "ROI-Selected : 左上(" << g_roi_rect.x << ", " << g_roi_rect.y << ") 幅" << g_roi_rect.width << " 高さ" << g_roi_rect.height << endl;
            mouse_position = Point(-1, -1); // ROI確定後はマウス位置をリセット
        }
    } else if (event == EVENT_MOUSEMOVE && g_roi_selecting) {
        mouse_position = Point(x, y);
    }
}


// Chamfer Matchingを実行する関数
Mat chamferMatch(const Mat& image_dist, const Mat& templ_edge) {
    int result_cols = image_dist.cols - templ_edge.cols + 1;
    int result_rows = image_dist.rows - templ_edge.rows + 1;
    
    // テンプレートが画像より大きい場合はマッチング不可
    if (result_cols <= 0 || result_rows <= 0) {
        return Mat::zeros(1, 1, CV_32F);
    }

    Mat result(result_rows, result_cols, CV_32F, Scalar(numeric_limits<float>::max())); // 結果を格納するMatを初期化

    // テンプレートのエッジ画素座標を抽出
    vector<Point> templ_edge_points;
    for (int y = 0; y < templ_edge.rows; ++y) {
        for (int x = 0; x < templ_edge.cols; ++x) {
            if (templ_edge.at<uchar>(y, x) > 0) {
                templ_edge_points.push_back(Point(x, y));
            }
        }
    }

    // 各位置でChamfer Matchingスコアを計算
    for (int y = 0; y < result_rows; ++y) {
        for (int x = 0; x < result_cols; ++x) {
            float current_score = 0;
            // テンプレートのエッジ画素ごとに、フレームの距離変換画像上の対応する位置の距離値を取得し合計
            for (const auto& p : templ_edge_points) {
                int img_x = x + p.x;
                int img_y = y + p.y;

                // 範囲チェック (image_distは通常フレーム全体をカバーするため不要な場合が多いが念のため)
                if (img_x >= 0 && img_x < image_dist.cols && img_y >= 0 && img_y < image_dist.rows) {
                    current_score += image_dist.at<float>(img_y, img_x);
                } else {
                    // テンプレートが画像範囲外に出る場合は大きな値を加算
                    current_score += numeric_limits<float>::max() / 100.0f; 
                }
            }
            result.at<float>(y, x) = current_score;
        }
    }
    return result;
}

bool setup_capture_and_window(VideoCapture& cap, const string& video_path, const string& window_name) {
    cap.open(video_path);
    if (!cap.isOpened()) return false;

    namedWindow(window_name);
    setMouseCallback(window_name, on_mouse, NULL);
    return true;
}

bool select_roi(VideoCapture& cap, const string& window_name) {
    cap >> g_initial_frame;
    if (g_initial_frame.empty()) return false;

    // ROI選択が完了するまでループ
    while (g_click_count < 2) {
        Mat display_frame = g_initial_frame.clone(); // 表示用に元のフレームをコピー

        if (g_roi_selecting && mouse_position.x != -1) {
            // ROI選択中で、マウスが動いている場合
            rectangle(display_frame, g_start_point, mouse_position, Scalar(0, 255, 0), 2);
            putText(display_frame, "Click again to select ROI", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        } else if (g_roi_selecting && g_click_count == 1) {
            // ROI選択中だが、まだマウスが動いていない（1回目のクリック直後）
            rectangle(display_frame, g_start_point, g_start_point, Scalar(0, 255, 0), 2); // 開始点に小さな矩形
            putText(display_frame, "Click again to select ROI", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        } else {
            // まだクリックしていない場合
            putText(display_frame, "Click twice to select ROI", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        }

        imshow(window_name, display_frame);

        int key = waitKey(1); 
        if (key == 'q' || key == 27) {
            return false;
        }
    }
    return true; 
}

// 選択されたROIからテンプレート画像を初期化する関数
bool initialize_template() {
    if (g_roi_rect.width <= 0 || g_roi_rect.height <= 0 ||
        g_roi_rect.x < 0 || g_roi_rect.y < 0 ||
        g_roi_rect.x + g_roi_rect.width > g_initial_frame.cols ||
        g_roi_rect.y + g_roi_rect.height > g_initial_frame.rows) {
        cerr << "選択されたROIが不正です。" << endl;
        return false;
    }
    
    // 元のRGB画像をテンプレートとして保存
    template_image_rgb = g_initial_frame(g_roi_rect).clone();
    g_original_template_rgb = template_image_rgb.clone(); // オリジナルをコピーして保持
    
    // グレースケール変換
    Mat gray_template;
    cvtColor(template_image_rgb, gray_template, COLOR_BGR2GRAY);

    // エッジ検出 (Canny): 閾値は調整が必要です。低いとエッジが増え、高いと減ります。
    Canny(gray_template, template_image_edge, 100, 200); 

    // 距離変換: DIST_MASK_5 は DIST_MASK_PRECISE より高速だが精度は若干落ちる
    distanceTransform(template_image_edge, template_image_dist, DIST_L2, DIST_MASK_5);

    // 初期テンプレートの表示
    imshow("Original Template RGB", g_original_template_rgb);
    imshow("Template Edge", template_image_edge);
    Mat display_dist;
    normalize(template_image_dist, display_dist, 0, 255, NORM_MINMAX, CV_8U);
    imshow("Template Distance Transform", display_dist);

    cout << "テンプレート画像が初期化されました (RGB, エッジ, 距離変換)。" << endl;
    // 初期テンプレートのROIをg_last_tracked_rectに設定
    g_last_tracked_rect = g_roi_rect; 
    return true;
}

// 各フレームで追跡処理を行う関数
// 成功した場合はtrue, 失敗または終了の場合はfalseを返す
bool process_tracking_frame(Mat& frame, const string& window_name) {    
    Mat display_frame = frame.clone();
    
    if (template_image_rgb.empty() || template_image_rgb.cols <= 0 || template_image_rgb.rows <= 0) {
        return false;
    }

    // フレームのグレースケール変換とエッジ検出
    Mat gray_frame;
    cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
    Mat frame_edge;
    Canny(gray_frame, frame_edge, 100, 200);

    // フレームの距離変換
    Mat frame_dist;
    distanceTransform(frame_edge, frame_dist, DIST_L2, DIST_MASK_5); // テンプレートと同じマスクサイズを使用

    Point matchLoc;

    // === 高速化オプション1: 探索範囲の限定 ===
    int margin = 50; // 探索マージン (ピクセル)
    Rect search_roi;
    if (g_last_tracked_rect.width > 0) { // 前のフレームで追跡成功していれば
        search_roi = g_last_tracked_rect;
        search_roi.x -= margin; 
        search_roi.y -= margin;
        search_roi.width += 2 * margin;
        search_roi.height += 2 * margin;
        // 画像範囲内にクリップ
        search_roi &= Rect(0, 0, frame_dist.cols, frame_dist.rows);
    } else {
        // 初回または追跡失敗時は全体を探索
        search_roi = Rect(0, 0, frame_dist.cols, frame_dist.rows);
    }

    if (search_roi.width <= 0 || search_roi.height <= 0) {
        return false; // 探索範囲が不正なら追跡を継続できない
    }

    // 探索範囲を切り抜く
    Mat sub_frame_dist = frame_dist(search_roi);
    
    // Chamfer Matchingを実行
    Mat chamfer_result = chamferMatch(sub_frame_dist, template_image_edge);
    
    if (chamfer_result.empty() || chamfer_result.cols <= 0 || chamfer_result.rows <= 0) {
        return false;
    }

    // 最も一致する位置を探す (最小値)
    double minVal, maxVal;
    Point minLoc_sub, maxLoc_sub;
    minMaxLoc(chamfer_result, &minVal, &maxVal, &minLoc_sub, &maxLoc_sub);
    
    // sub_frame_dist 上の座標を、元のフレームの座標に変換
    matchLoc = minLoc_sub + search_roi.tl();
    // === 高速化オプション1 終わり ===

    // 追跡結果の矩形を描画
    Rect tracked_rect(matchLoc.x, matchLoc.y, template_image_rgb.cols, template_image_rgb.rows); 
    rectangle(display_frame, tracked_rect, Scalar(0, 0, 255), 2);

    // 重心位置を計算して表示
    Point center_point(tracked_rect.x + tracked_rect.width / 2, tracked_rect.y + tracked_rect.height / 2);
    circle(display_frame, center_point, 5, Scalar(0, 255, 0), -1); // 緑色の点で重心表示
    putText(display_frame, "Center: (" + to_string(center_point.x) + ", " + to_string(center_point.y) + ")",
            Point(center_point.x + 10, center_point.y + 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    putText(display_frame, "Score: " + to_string(static_cast<int>(minVal)), Point(center_point.x + 10, center_point.y + 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

    // テンプレートの逐次更新
    // 新しいフレームで最もよく一致した領域を次のテンプレートとする
    // ただし、追跡矩形が画面外に出ないようにチェック
    if (tracked_rect.width > 0 && tracked_rect.height > 0 &&
        tracked_rect.x >= 0 && tracked_rect.y >= 0 &&
        tracked_rect.x + tracked_rect.width <= frame.cols &&
        tracked_rect.y + tracked_rect.height <= frame.rows) {
        
        template_image_rgb = frame(tracked_rect).clone();
        
        Mat gray_template_next;
        cvtColor(template_image_rgb, gray_template_next, COLOR_BGR2GRAY);
        Canny(gray_template_next, template_image_edge, 100, 200); // 同じ閾値を使用
        distanceTransform(template_image_edge, template_image_dist, DIST_L2, DIST_MASK_5); // 同じマスクサイズを使用

        // 逐次更新されたテンプレートを表示
        imshow("Updated Template RGB", template_image_rgb);
        Mat display_dist_next;
        normalize(template_image_dist, display_dist_next, 0, 255, NORM_MINMAX, CV_8U);
        imshow("Updated Template Distance Transform", display_dist_next);
        
        // 追跡成功時にg_last_tracked_rectを更新
        g_last_tracked_rect = tracked_rect; 
    } else {
        //追跡矩形が画面外に出たか不正なサイズの場合
        cerr << "追跡矩形が画面外に出たか、不正なサイズになりました。追跡を終了します。" << endl;
        return false;
    }

    imshow(window_name, display_frame);

    int key = waitKey(1);
    if (key == 'q' || key == 27) {
        return false;
    }
    return true;
}

// ==================== main 関数 ====================
int main() {
    // =============== 1.　動画の読み込み =================
    string video_path = "traffic.mov";
    string window_name = "Camera Feed";

    VideoCapture cap;
    if (!setup_capture_and_window(cap, video_path, window_name)) {
        cerr << "動画ファイルを開けませんでした: " << video_path << endl;
        return -1;
    }

    // =============== 2.　ROI選択 =================
    cout << "動画からROIを選択してください。オブジェクトを囲むように2回クリックしてください。" << endl;
    if (!select_roi(cap, window_name)) {
        cout << "ROI選択がキャンセルされました。" << endl;
        cap.release();
        destroyAllWindows();
        return 0;
    }

    // =============== 3.　テンプレートの初期化 =================
    if (!initialize_template()) {
        cap.release();
        destroyAllWindows();
        return -1;
    }

    // ============== 4. トラッキング =================
    Mat frame;
    cout << "追跡を開始します。'q' または 'Esc' キーで終了します。" << endl;
    while (true) {
        cap >> frame; 

        if (frame.empty()) break;
        if (!process_tracking_frame(frame, window_name)) break;
    }

    cap.release();
    destroyAllWindows();

    cout << "追跡が終了しました。" << endl;

    return 0;
}