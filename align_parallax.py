import cv2
import numpy as np
import argparse

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

    # 特徴点検出（ORB）
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # 特徴点マッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 上位マッチのみ使用
    good_matches = matches[:int(len(matches) * 0.3)]

    if len(good_matches) < 10:
        print("警告: マッチング点が少ないです")

    # マッチング点の座標を取得
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # RANSACで外れ値を除去してアフィン変換を推定
    M_affine, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC,
                                                     ransacReprojThreshold=5.0)

    if M_affine is None:
        print("警告: アフィン変換の推定に失敗しました。垂直シフトのみ適用します")
        y_diffs = pts2[:, 1] - pts1[:, 1]
        median_y_diff = np.median(y_diffs)
        h, w = img2.shape[:2]
        M = np.float32([[1, 0, 0], [0, 1, -median_y_diff]])
    else:
        # 回転角度を計算
        angle = np.arctan2(M_affine[1, 0], M_affine[0, 0]) * 180 / np.pi

        # スケールを計算
        scale_x = np.sqrt(M_affine[0, 0]**2 + M_affine[1, 0]**2)
        scale_y = np.sqrt(M_affine[0, 1]**2 + M_affine[1, 1]**2)

        print(f"検出された回転角度: {angle:.2f}度")
        print(f"検出されたスケール: X={scale_x:.4f}, Y={scale_y:.4f}")
        print(f"検出された平行移動: X={M_affine[0, 2]:.2f}, Y={M_affine[1, 2]:.2f}ピクセル")

        # パララックスを保持するため、X方向の移動を制限
        # 回転とY方向の移動のみ適用
        M = M_affine.copy()
        # X方向の移動を元に戻す（パララックスを保持）
        M[0, 2] = 0

    # 画像2を補正
    h, w = img2.shape[:2]
    img2_aligned = cv2.warpAffine(img2, M, (w, h))

    # 横並びに結合
    combined = np.hstack([img1, img2_aligned])

    # 結果を保存
    cv2.imwrite(output1_path, img1)
    cv2.imwrite(output2_path, img2_aligned)

    # 結合画像も保存
    combined_path = output1_path.rsplit('.', 1)[0] + '_combined.' + output1_path.rsplit('.', 1)[1]
    cv2.imwrite(combined_path, combined)

    print(f"\n補正完了:")
    print(f"  左画像: {output1_path}")
    print(f"  右画像（補正済み）: {output2_path}")
    print(f"  結合画像: {combined_path}")

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
