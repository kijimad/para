import cv2
import numpy as np
import argparse
import os
from datetime import datetime
from PIL import Image

def align_images(img1_path, img2_path, output1_path, output2_path):
    """
    2つの視点画像の高さと角度を補正する（パララックス用）
    """
    # 画像を読み込む
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise ValueError("画像を読み込めませんでした")

    # グレースケールに変換
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 特徴点検出（AKAZE - ORBより精度が高い）
    detector = cv2.AKAZE_create()
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    print(f"検出された特徴点: 左={len(kp1)}, 右={len(kp2)}")

    # 特徴点マッチング（knnMatchでLowe's ratio testを適用）
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test（誤マッチを除去）
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:  # 0.7が一般的な閾値
                good_matches.append(m)

    print(f"Lowe's ratio test後: {len(good_matches)}個のマッチ")

    if len(good_matches) < 10:
        print("警告: マッチング点が少ないです")

    # マッチング点の座標を取得
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Y方向のズレのみを計算（視差画像はX方向の差があるのが正常）
    y_diffs = pts2[:, 1] - pts1[:, 1]

    # 外れ値除去: 中央値から3σ以上離れた点を除外
    median_y = np.median(y_diffs)
    std_y = np.std(y_diffs)
    threshold = 3 * std_y
    mask = np.abs(y_diffs - median_y) < threshold
    y_diffs_filtered = y_diffs[mask]

    print(f"マッチング点: {len(y_diffs)}個 → 外れ値除去後: {len(y_diffs_filtered)}個")

    # フィルタ後の中央値を計算
    median_y_shift = np.median(y_diffs_filtered)

    print(f"検出されたY方向のズレ: {median_y_shift:.2f}ピクセル")
    print(f"Y方向のズレの標準偏差: {np.std(y_diffs_filtered):.2f}ピクセル")

    # Y方向の平行移動のみ適用（回転・スケールは適用しない）
    M = np.float32([[1, 0, 0], [0, 1, -median_y_shift]])

    # 画像2を補正
    h, w = img2.shape[:2]
    img2_aligned = cv2.warpAffine(img2, M, (w, h))

    # 横並びに結合
    combined = np.hstack([img1, img2_aligned])

    # 入力ファイル名から拡張子を除いたベース名を取得
    input1_basename = os.path.splitext(os.path.basename(img1_path))[0]
    input2_basename = os.path.splitext(os.path.basename(img2_path))[0]
    prefix = f"{input1_basename}_{input2_basename}"

    # 出力パスに入力ファイル名を先頭に挿入
    if '.' in output1_path:
        base1, ext1 = output1_path.rsplit('.', 1)
        dir1 = os.path.dirname(base1)
        file1 = os.path.basename(base1)
        if dir1:
            output1_path_with_time = f"{dir1}/{prefix}_{file1}.{ext1}"
        else:
            output1_path_with_time = f"{prefix}_{file1}.{ext1}"
    else:
        dir1 = os.path.dirname(output1_path)
        file1 = os.path.basename(output1_path)
        if dir1:
            output1_path_with_time = f"{dir1}/{prefix}_{file1}"
        else:
            output1_path_with_time = f"{prefix}_{file1}"

    if '.' in output2_path:
        base2, ext2 = output2_path.rsplit('.', 1)
        dir2 = os.path.dirname(base2)
        file2 = os.path.basename(base2)
        if dir2:
            output2_path_with_time = f"{dir2}/{prefix}_{file2}.{ext2}"
        else:
            output2_path_with_time = f"{prefix}_{file2}.{ext2}"
    else:
        dir2 = os.path.dirname(output2_path)
        file2 = os.path.basename(output2_path)
        if dir2:
            output2_path_with_time = f"{dir2}/{prefix}_{file2}"
        else:
            output2_path_with_time = f"{prefix}_{file2}"

    # 結果を保存
    cv2.imwrite(output1_path_with_time, img1)
    cv2.imwrite(output2_path_with_time, img2_aligned)

    # 結合画像も保存
    if dir1:
        combined_path = f"{dir1}/{prefix}_{file1}_combined.{ext1}"
    else:
        combined_path = f"{prefix}_{file1}_combined.{ext1}"
    cv2.imwrite(combined_path, combined)

    # GIFアニメーションを作成
    if dir1:
        gif_path = f"{dir1}/{prefix}_{file1}_animation.gif"
    else:
        gif_path = f"{prefix}_{file1}_animation.gif"

    # OpenCVのBGRをRGBに変換してPIL Imageに変換
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_aligned_rgb = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB)

    pil_img1 = Image.fromarray(img1_rgb)
    pil_img2 = Image.fromarray(img2_aligned_rgb)

    # 画像をリサイズ（ファイルサイズを減らすため）
    max_width = 2000
    if pil_img1.width > max_width:
        ratio = max_width / pil_img1.width
        new_size = (max_width, int(pil_img1.height * ratio))
        pil_img1 = pil_img1.resize(new_size, Image.Resampling.LANCZOS)
        pil_img2 = pil_img2.resize(new_size, Image.Resampling.LANCZOS)

    # GIFとして保存
    # 左→右のループ
    pil_img1.save(
        gif_path,
        format='GIF',
        save_all=True,
        append_images=[pil_img2],
        duration=200,
        loop=0
    )

    print(f"\n補正完了:")
    print(f"  左画像: {output1_path_with_time}")
    print(f"  右画像（補正済み）: {output2_path_with_time}")
    print(f"  結合画像: {combined_path}")
    print(f"  GIFアニメーション: {gif_path}")

    return M

def main():
    parser = argparse.ArgumentParser(
        description='2視点画像の高さを補正してパララックス画像を作成'
    )
    parser.add_argument('img1', help='左側画像のパス')
    parser.add_argument('img2', help='右側画像のパス')
    parser.add_argument('--output1', default='left_aligned.jpg',
                       help='出力画像1のパス（デフォルト: left_aligned.jpg）')
    parser.add_argument('--output2', default='right_aligned.jpg',
                       help='出力画像2のパス（デフォルト: right_aligned.jpg）')

    args = parser.parse_args()

    align_images(args.img1, args.img2, args.output1, args.output2)

if __name__ == '__main__':
    main()
