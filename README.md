# univImageProcessing

本システムの動作環境は以下の通りである．

• OS：macOS Sequoia
• Chip：Apple M2
• Camera：mac book air M2 2022 標準搭載のもの
• Memory：16GB
• 開発⾔語：C++
• 主要ライブラリ：OpenCV4
• 開発ツール：Visual Studio Code ( Visual Studio Code 2022 ではない )
• ビルド方法： g++ exp2ac.cpp -o main `pkg-config --cflags --libs opencv4`
• ディレクトリ構成
    .
    ├── expFull.cpp
    └── wallpaperImage.jpg

exp1：カメラからの入力映像から手を認識させるシステム
クロマキー合成を実装しているため， "wallpaperImage.png" という名前で背景画像を保存する必要がある．

exp2：テンプレートマッチングを行い，対象物体の画像上の変位を計算する簡易的なシステム
入力映像が必要であるので，"traffic.mov" と言う名前で入力映像を保存する必要がある．（交通道路の映像）