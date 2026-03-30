import cv2
import os
import numpy as np
from tqdm import tqdm  # 导入进度条库

# ================= 配置区域 =================
INPUT_DIR = "/root/autodl-tmp/jigsawas/Suturing"      # 存放这10个视频的文件夹路径
OUTPUT_DIR = "/root/autodl-tmp/SV_test/data/Suturing/heichole_frames"     # 存放抽帧结果的总文件夹路径
TARGET_FRAMES = 720                       # 每个视频需要提取的帧数
WHITE_THRESHOLD = 250                     # 判断白屏的亮度阈值（0-255，越接近255越白）
# ============================================

def process_videos():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    video_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    total_videos = len(video_files)
    
    for video_idx, video_name in enumerate(video_files):
        video_path = os.path.join(INPUT_DIR, video_name)
        video_out_dir = os.path.join(OUTPUT_DIR, os.path.splitext(video_name)[0])
        os.makedirs(video_out_dir, exist_ok=True)
        
        # 打印整体视频处理进度
        print(f"\n[{video_idx + 1}/{total_videos}] 正在处理视频: {video_name} ...")
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频总帧数，用于进度条计算
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        valid_frame_indices = []
        
        # --- 第一步：扫描视频，找出所有有效帧 ---
        # 使用 tqdm 包装 range() 来生成进度条
        for frame_idx in tqdm(range(total_frames), desc="  -> 扫描有效帧", unit="帧"):
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < WHITE_THRESHOLD:
                valid_frame_indices.append(frame_idx)
                
        total_valid = len(valid_frame_indices)
        print(f"  -> 扫描完毕。剔除白屏后，剩余有效帧数: {total_valid}")
        
        if total_valid < TARGET_FRAMES:
            print(f"  [警告] 视频 {video_name} 的有效帧数（{total_valid}）少于 {TARGET_FRAMES} 帧，跳过保存。")
            cap.release()
            continue
            
        # --- 第二步：在有效帧中均匀挑出 720 个序号 ---
        selected_indices = np.linspace(0, total_valid - 1, TARGET_FRAMES, dtype=int)
        target_frame_numbers = [valid_frame_indices[i] for i in selected_indices]
        
        # --- 第三步：跳转并保存这 720 帧 ---
        # 使用 tqdm 包装 enumerate 来生成保存进度的进度条
        for count, target_idx in tqdm(enumerate(target_frame_numbers), total=TARGET_FRAMES, desc="  -> 抽取并保存", unit="张"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if ret:
                # 命名格式从 frame_0000.jpg 开始
                out_path = os.path.join(video_out_dir, f"frame_{count:04d}.jpg")
                cv2.imwrite(out_path, frame)
                
        cap.release()

if __name__ == "__main__":
    process_videos()
    print("\n🎉 所有视频批量抽帧完毕！")