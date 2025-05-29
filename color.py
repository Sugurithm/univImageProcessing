# 色調領域を表示するためのプログラム

import cv2
import numpy as np

# --- 指定したHSV範囲 ---
lower_hsv = np.array([5, 50, 60])
upper_hsv = np.array([20, 150, 255])

# スペクトラム画像の幅と高さの定義
image_width = 800  # 画像の幅
image_height = 200 # 画像の高さ（この高さにグラデーションが表示される）

# 空の画像を用意（BGR形式なので3チャンネル）
spectrum_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# H, S, Vの各チャンネルでステップ数を決定
# グラデーションを滑らかにするため細かくする
h_steps = (upper_hsv[0] - lower_hsv[0]) + 1
s_steps = (upper_hsv[1] - lower_hsv[1]) + 1
v_steps = (upper_hsv[2] - lower_hsv[2]) + 1

spectrum_colors_hsv = []
for h_val in range(lower_hsv[0], upper_hsv[0] + 1):
    for s_val in range(lower_hsv[1], upper_hsv[1] + 1, 10):
        for v_val in range(lower_hsv[2], upper_hsv[2] + 1, 10):
            h_clamped = np.clip(h_val, 0, 179)
            s_clamped = np.clip(s_val, 0, 255)
            v_clamped = np.clip(v_val, 0, 255)
            spectrum_colors_hsv.append([h_clamped, s_clamped, v_clamped])

spectrum_colors_hsv_np = np.array(spectrum_colors_hsv, dtype=np.uint8).reshape(-1, 1, 3)
spectrum_colors_bgr = cv2.cvtColor(spectrum_colors_hsv_np, cv2.COLOR_HSV2BGR)
total_generated_colors = len(spectrum_colors_bgr)


for x in range(image_width):
    color_idx = int((x / (image_width - 1)) * (total_generated_colors - 1))
    spectrum_image[:, x] = spectrum_colors_bgr[color_idx][0]

# 表示
cv2.imshow("HSV Color Spectrum", spectrum_image)
cv2.waitKey(0)
cv2.destroyAllWindows()