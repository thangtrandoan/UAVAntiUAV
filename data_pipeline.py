import os
import glob
import json
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Data Pipeline for UAV-Anti-UAV ReID")
    parser.add_argument("--data-dir", type=str, default="../UAV-Anti-UAV", help="Path to raw dataset")
    parser.add_argument("--output-dir", type=str, default="./processed", help="Path to output processed data")
    parser.add_argument("--num-before-frames", type=int, default=16, help="Number of frames before disappearance")
    parser.add_argument("--num-after-frames", type=int, default=16, help="Number of frames after reappearance")
    parser.add_argument("--bbox-padding", type=float, default=0.2, help="Padding ratio for bounding box")
    parser.add_argument("--crop-size", type=int, default=256, help="Size to resize the cropped image")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of processes to use")
    return parser.parse_args()

def crop_and_resize(frame, bbox, padding, crop_size):
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    
    # Add padding
    pad_w = bw * padding
    pad_h = bh * padding
    
    x1 = int(max(0, x - pad_w))
    y1 = int(max(0, y - pad_h))
    x2 = int(min(w, x + bw + pad_w))
    y2 = int(min(h, y + bh + pad_h))
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    crop = frame[y1:y2, x1:x2]
    try:
        resized = cv2.resize(crop, (crop_size, crop_size))
        return resized
    except Exception as e:
        return None

def process_sequence(seq_path, output_base, split, args):
    seq_name = os.path.basename(seq_path)
    
    video_path = os.path.join(seq_path, f"{seq_name}.mp4")
    gt_path = os.path.join(seq_path, "groundtruth_rect.txt")
    absent_path = os.path.join(seq_path, "absent.txt")
    attr_path = os.path.join(seq_path, "attributes.txt")
    lang_path = os.path.join(seq_path, "language.txt")
    
    if not all(os.path.exists(p) for p in [video_path, gt_path, absent_path]):
        return {"status": "error", "message": f"Missing files in {seq_path}"}
        
    # Read files
    try:
        with open(gt_path, "r") as f:
            bboxes = []
            for line in f:
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 4:
                    bboxes.append([int(float(p)) for p in parts[:4]])
                else:
                    bboxes.append([0, 0, 0, 0])
                    
        with open(absent_path, "r") as f:
            absent = [int(line.strip()) for line in f if line.strip().isdigit()]
            
        attributes = []
        if os.path.exists(attr_path):
            with open(attr_path, "r") as f:
                for line in f:
                    try:
                        attributes.append(int(line.strip()))
                    except ValueError:
                        pass
        
        language = ""
        if os.path.exists(lang_path):
            with open(lang_path, "r") as f:
                language = f.read().strip()
                
    except Exception as e:
        return {"status": "error", "message": f"Error reading txt files in {seq_path}: {e}"}

    # Find transition segments
    events = []
    in_disappearance = False
    disappear_start = -1
    
    for i in range(len(absent)):
        if absent[i] == 1 and not in_disappearance:
            in_disappearance = True
            disappear_start = i
        elif absent[i] == 0 and in_disappearance:
            in_disappearance = False
            disappear_end = i
            events.append((disappear_start, disappear_end))
            
    if not events:
        return {"status": "skipped", "message": f"No disappearance in {seq_path}", "seq_name": seq_name}
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return {"status": "error", "message": f"Could not read video {video_path}"}
        
    pairs = []
    
    for event_idx, (start_idx, end_idx) in enumerate(events):
        before_start = max(0, start_idx - args.num_before_frames)
        before_end = start_idx
        
        after_start = end_idx
        after_end = min(total_frames, end_idx + args.num_after_frames)
        
        needed_frames = list(range(before_start, before_end)) + list(range(after_start, after_end))
        needed_frames = sorted(list(set(needed_frames)))
        
        if not needed_frames:
            continue
            
        event_out_dir = os.path.join(output_base, split, f"{seq_name}_event_{event_idx}")
        before_dir = os.path.join(event_out_dir, "before")
        after_dir = os.path.join(event_out_dir, "after")
        os.makedirs(before_dir, exist_ok=True)
        os.makedirs(after_dir, exist_ok=True)
        
        before_frames_files = []
        after_frames_files = []
        
        # Read frames
        for frame_idx in needed_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            if frame_idx >= len(bboxes):
                continue
                
            bbox = bboxes[frame_idx]
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
                
            crop = crop_and_resize(frame, bbox, args.bbox_padding, args.crop_size)
            if crop is None:
                continue
                
            frame_name = f"frame_{frame_idx:04d}.jpg"
            if frame_idx < start_idx:
                out_path = os.path.join(before_dir, frame_name)
                cv2.imwrite(out_path, crop)
                before_frames_files.append(frame_name)
            else:
                out_path = os.path.join(after_dir, frame_name)
                cv2.imwrite(out_path, crop)
                after_frames_files.append(frame_name)
                
        if before_frames_files and after_frames_files:
            pairs.append({
                "sequence_id": seq_name,
                "event_index": event_idx,
                "identity_id": None, # Will be assigned later globally
                "before_frames": before_frames_files,
                "after_frames": after_frames_files,
                "disappearance_duration_frames": end_idx - start_idx,
                "language_description": language,
                "attributes": attributes
            })
            
    cap.release()
    
    return {
        "status": "success",
        "seq_name": seq_name,
        "pairs": pairs,
        "attributes": attributes if pairs else []
    }

def main():
    args = parse_args()
    
    train_dir = os.path.join(args.data_dir, "Train")
    test_dir = os.path.join(args.data_dir, "Test")
    
    if not os.path.exists(train_dir) and not os.path.exists(test_dir):
        print(f"Warning: {args.data_dir} does not contain Train or Test directories.")

    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)
    
    stats = {
        "processed_seqs": 0,
        "skipped_seqs": 0,
        "error_seqs": 0,
        "total_pairs": 0,
        "durations": [],
        "attr_counts": defaultdict(int)
    }
    
    all_pairs_train = []
    all_pairs_test = []
    
    def process_split(split_name, split_dir, out_pairs_list):
        if not os.path.exists(split_dir):
            return
            
        seqs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        
        tasks = []
        for seq_name in seqs:
            seq_path = os.path.join(split_dir, seq_name)
            tasks.append((seq_path, args.output_dir, split_name, args))
            
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_sequence, *task): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Processing {split_name}"):
                res = future.result()
                
                if res["status"] == "error":
                    stats["error_seqs"] += 1
                    print(f"Error: {res['message']}")
                elif res["status"] == "skipped":
                    stats["skipped_seqs"] += 1
                elif res["status"] == "success":
                    if res["pairs"]:
                        stats["processed_seqs"] += 1
                        out_pairs_list.extend(res["pairs"])
                        stats["total_pairs"] += len(res["pairs"])
                        
                        for p in res["pairs"]:
                            stats["durations"].append(p["disappearance_duration_frames"])
                            
                        # Attribute distribution
                        attrs = res["attributes"]
                        for i, a in enumerate(attrs):
                            if a == 1:
                                stats["attr_counts"][i] += 1
                    else:
                        stats["skipped_seqs"] += 1

    process_split("train", train_dir, all_pairs_train)
    process_split("test", test_dir, all_pairs_test)
    
    # Reassign identity_id globally from 0 to N-1
    unique_seqs = set()
    for p in all_pairs_train + all_pairs_test:
        unique_seqs.add(p["sequence_id"])
        
    seq_to_id = {seq: i for i, seq in enumerate(sorted(list(unique_seqs)))}
    
    for p in all_pairs_train:
        p["identity_id"] = seq_to_id[p["sequence_id"]]
        
    for p in all_pairs_test:
        p["identity_id"] = seq_to_id[p["sequence_id"]]
    
    # Save metadata
    with open(os.path.join(args.output_dir, "pairs_train.json"), "w") as f:
        json.dump(all_pairs_train, f, indent=4)
        
    with open(os.path.join(args.output_dir, "pairs_test.json"), "w") as f:
        json.dump(all_pairs_test, f, indent=4)
        
    # Print stats
    print("\n" + "="*50)
    print("DATA PIPELINE REPORT")
    print("="*50)
    print(f"Total sequences processed successfully : {stats['processed_seqs']}")
    print(f"Total sequences skipped (no disappearance) : {stats['skipped_seqs']}")
    print(f"Total sequences with errors            : {stats['error_seqs']}")
    print(f"Total unique identities (drones)       : {len(unique_seqs)}")
    print(f"Total pairs (T_before, T_after) created: {stats['total_pairs']}")
    
    if stats["durations"]:
        avg_dur = np.mean(stats["durations"])
        min_dur = np.min(stats["durations"])
        max_dur = np.max(stats["durations"])
        print(f"\nDisappearance Duration Stats (frames):")
        print(f"  - Average : {avg_dur:.2f}")
        print(f"  - Min     : {min_dur}")
        print(f"  - Max     : {max_dur}")
        
    print("\nAttribute Distribution (Videos with feature == 1):")
    # Mapping for common attributes in tracking datasets
    attr_names = {
        0: "Fast Motion", 1: "Background Clutter", 2: "Out-of-View",
        3: "Illumination Variation", 4: "Viewpoint Change", 5: "Scale Variation",
        6: "Deformation", 7: "Occlusion", 8: "Motion Blur", 9: "Low Resolution",
        10: "Camera Motion", 11: "Thermal Crossover", 12: "Night-time",
        13: "Fake UAV", 14: "Zoom"
    }
    
    for i in sorted(stats["attr_counts"].keys()):
        count = stats["attr_counts"][i]
        name = attr_names.get(i, f"Attribute_{i}")
        print(f"  - {name:<25}: {count}")
    print("="*50)

if __name__ == "__main__":
    main()
