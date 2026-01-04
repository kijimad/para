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

    # 画像の高さを取得
    h, w = img2.shape[:2]

    # Y方向のズレの傾向から回転角度を推定（より正確）
    # 画像の左側と右側でY方向のズレの差を計算
    w_img = img1.shape[1]

    # 左側1/3の特徴点
    mask_left = pts1[:, 0] < w_img / 3
    pts1_left = pts1[mask_left]
    pts2_left = pts2[mask_left]

    # 右側1/3の特徴点
    mask_right = pts1[:, 0] > 2 * w_img / 3
    pts1_right = pts1[mask_right]
    pts2_right = pts2[mask_right]

    if len(pts1_left) > 10 and len(pts1_right) > 10:
        # 左側と右側のY方向のズレの中央値を計算
        y_diff_left = np.median(pts2_left[:, 1] - pts1_left[:, 1])
        y_diff_right = np.median(pts2_right[:, 1] - pts1_right[:, 1])

        # Y方向のズレの差から回転角度を計算
        # tan(angle) = (y_diff_right - y_diff_left) / width
        delta_y = y_diff_right - y_diff_left
        angle = np.arctan2(delta_y, w_img) * 180 / np.pi

        print(f"左側Y方向のズレ: {y_diff_left:.2f}px, 右側Y方向のズレ: {y_diff_right:.2f}px")
        print(f"検出された回転角度: {angle:.4f}度")
    else:
        # フォールバック: アフィン変換で推定
        M_affine, inliers = cv2.estimateAffinePartial2D(
            pts2, pts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            confidence=0.99,
            maxIters=2000
        )

        if M_affine is not None:
            angle = np.arctan2(M_affine[1, 0], M_affine[0, 0]) * 180 / np.pi
            print(f"検出された回転角度（アフィン変換）: {angle:.4f}度")
        else:
            angle = 0
            print(f"回転検出失敗、角度 = 0度")

    # 回転角度が小さい場合は回転補正をスキップ
    if abs(angle) < 0.2:
        print(f"回転角度が小さい（{abs(angle):.4f}度 < 0.2度）ため、回転補正をスキップ")
        img2_rotated = img2.copy()
        pts2_rotated = pts2.copy()
    else:
        print(f"回転補正を適用（左端中央を軸に回転）")
        # 回転を補正（左端中央を軸にする）
        center = (0, h / 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        img2_rotated = cv2.warpAffine(img2, M_rot, (w, h))

        # 特徴点も回転
        pts2_homogeneous = np.hstack([pts2, np.ones((len(pts2), 1))])
        pts2_rotated = (M_rot @ pts2_homogeneous.T).T[:, :2]

    # Y方向のズレを計算（回転後の特徴点を使用）
    y_diffs = pts2_rotated[:, 1] - pts1[:, 1]

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

    # Y方向の平行移動のみ適用
    M_translate = np.float32([[1, 0, 0], [0, 1, -median_y_shift]])
    img2_aligned = cv2.warpAffine(img2_rotated, M_translate, (w, h))

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

    # アニメーションファイルのパスを設定
    if dir1:
        gif_path = f"{dir1}/{prefix}_{file1}_animation.gif"
        webp_path = f"{dir1}/{prefix}_{file1}_animation.webp"
    else:
        gif_path = f"{prefix}_{file1}_animation.gif"
        webp_path = f"{prefix}_{file1}_animation.webp"

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
    pil_img1.save(
        gif_path,
        format='GIF',
        save_all=True,
        append_images=[pil_img2],
        duration=200,
        loop=0
    )

    # WebPとして保存
    pil_img1.save(
        webp_path,
        format='WEBP',
        save_all=True,
        append_images=[pil_img2],
        duration=200,
        loop=0,
        lossless=False,
        quality=80,
        method=4
    )

    print(f"\n補正完了:")
    print(f"  左画像: {output1_path_with_time}")
    print(f"  右画像（補正済み）: {output2_path_with_time}")
    print(f"  結合画像: {combined_path}")
    print(f"  GIFアニメーション: {gif_path}")
    print(f"  WebPアニメーション: {webp_path}")

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
