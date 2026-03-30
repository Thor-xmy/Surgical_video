import json
import os

def process_meta_files():
    # 你的三个原始标签文件
    meta_files = [
        'meta_file_Knot_Tying.txt',
        'meta_file_Needle_Passing.txt',
        'meta_file_Suturing.txt'
    ]
    
    # 存放所有视频标签的字典
    annotations = {}
    
    for file_path in meta_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 警告: 未找到文件 {file_path}，已跳过。")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 按空白字符(空格或制表符)分割
                parts = line.split()
                
                # 根据 readme.txt 提取数据
                # parts[0]: 视频文件名 (例如 Suturing_B001)
                # parts[1]: 技能水平 self-proclaimed (E, I, N)
                # parts[2]: GRS总分 (技能水平打分)
                # parts[3:9]: 6个维度的细项得分
                
                if len(parts) >= 3:
                    video_id = parts[0]
                    skill_level = parts[1]
                    score = float(parts[2])  # 提取作为标签的 GRS 总分
                    
                    # 提取细项得分（如果有的话）
                    individual_scores = [float(x) for x in parts[3:9]] if len(parts) >= 9 else []
                    
                    # 组装符合 utils/data_loader_video_level_frames.py 要求的格式
                    # 必须包含 "score" 键
                    annotations[video_id] = {
                        "score": score,
                        "skill_level": skill_level,
                        "individual_scores": individual_scores
                    }
                    
        print(f"✅ 成功读取: {file_path}")

    # 输出文件路径 (与你 config 和 dataloader 默认参数对齐)
    output_file = 'annotations_combined.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入 JSON，设置 indent=4 使其格式化并易于人类阅读
        json.dump(annotations, f, indent=4)
        
    print(f"\n🎉 转换完成！共处理了 {len(annotations)} 个视频的标签。")
    print(f"📁 结果已保存至当前目录下的: {output_file}")

if __name__ == '__main__':
    process_meta_files()