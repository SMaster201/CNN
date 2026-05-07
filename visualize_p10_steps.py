import cv2
import numpy as np
import os

def main():
    # ---------------------------------------------------------
    # ⚙️ 設定區
    # ---------------------------------------------------------
    target_image_name = "101001.bmp"  # 您可以換成任何 img64x128 內的圖
    input_path = os.path.join("dataset", target_image_name)
    
    output_dir = "dataset\preprocessing"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 開始分析並視覺化切割與強化流程: {input_path}")
    
    # Step 0: 原始影像 (灰階讀取)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ 找不到圖片: {input_path}，請確認路徑")
        return
    cv2.imwrite(os.path.join(output_dir, "step0_original.png"), img)
    print("Step 0: 已儲存原始影像")

    # ---------------------------------------------------------
    # ✂️ 第一階段：視覺化切割過程 (Slice)
    # ---------------------------------------------------------
    h, w = img.shape[:2]
    row_bounds = [(0, int(h * 0.4)), (int(h * 0.3), int(h * 0.7)), (int(h * 0.6), h)]
    col_bounds = [(0, int(w * 0.6)), (int(w * 0.4), w)]

    # 畫出 3 列的切割線 (紅色)
    img_color_rows = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, (r_start, r_end) in enumerate(row_bounds, 1):
        cv2.rectangle(img_color_rows, (0, r_start), (w, r_end), (0, 0, 255), 1)
        # 實際把 3 列存出來
        row_img = img[r_start:r_end, :]
        cv2.imwrite(os.path.join(output_dir, f"step1_sliced_row_{i}.png"), row_img)
    cv2.imwrite(os.path.join(output_dir, "step1_cut_lines_3rows.png"), img_color_rows)
    print("Step 1: 已畫出 3 列切割線，並儲存列區塊圖")

    # 畫出 6 塊的切割線 (綠色)
    img_color_grid = img_color_rows.copy()
    target_block = None  # 用來抓取 r1_c1 作為後續示範
    
    for r_idx, (r_start, r_end) in enumerate(row_bounds, 1):
        for c_idx, (c_start, c_end) in enumerate(col_bounds, 1):
            cv2.rectangle(img_color_grid, (c_start, r_start), (c_end, r_end), (0, 255, 0), 1)
            
            # 實際把 6 塊存出來
            block_img = img[r_start:r_end, c_start:c_end]
            cv2.imwrite(os.path.join(output_dir, f"step2_sliced_r{r_idx}_c{c_idx}.png"), block_img)
            
            # 抓取左上角第一格作為 P10 示範
            if r_idx == 1 and c_idx == 1:
                target_block = block_img

    cv2.imwrite(os.path.join(output_dir, "step2_cut_lines_6blocks.png"), img_color_grid)
    print("Step 2: 已畫出 6 格切割線，並儲存 6 張單格小圖")

    # ---------------------------------------------------------
    # 🛠️ 第二階段：針對單格 (r1_c1) 進行 P10 強化流程
    # ---------------------------------------------------------
    print("\n👉 抽取左上角第一格 (r1_c1) 進入 P10 強化管線...")

    # Step 3: 影像放大 5 倍
    resized = cv2.resize(target_block, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(output_dir, "step3_p10_upscaled.png"), resized)
    print("Step 3: 已儲存單格放大影像 (5x)")

    # Step 4: 大津法二值化 (Otsu's Thresholding)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_dir, "step4_p10_binary.png"), thresh)
    print("Step 4: 已儲存單格二值化影像")

    # Step 5: 垂直膨脹 (Vertical Dilation)
    v_kernel = np.ones((7, 1), np.uint8)
    dilated = cv2.dilate(thresh, v_kernel, iterations=1)
    cv2.imwrite(os.path.join(output_dir, "step5_p10_dilated.png"), dilated)
    print("Step 5: 已儲存單格垂直膨脹影像")

    # Step 6: 黑白反轉與填邊
    inverted = cv2.bitwise_not(dilated)
    final = cv2.copyMakeBorder(inverted, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(os.path.join(output_dir, "step6_p10_final_padded.png"), final)
    print("Step 6: 已儲存單格反轉與填邊最終影像")

    print(f"\n✅ 全部處理完成！請至 '{output_dir}' 資料夾查看視覺化結果。")

if __name__ == "__main__":
    main()
