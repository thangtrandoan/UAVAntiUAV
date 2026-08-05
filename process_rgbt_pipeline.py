import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
from tqdm import tqdm

def crop_and_resize(frame, bbox, padding, crop_size):
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    
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

def process_sequence(seq_path, output_base, split, num_before_frames=16, num_after_frames=16, bbox_padding=0.2, crop_size=256):
    seq_name = os.path.basename(seq_path)
    
    video_path = os.path.join(seq_path, "visible.mp4")
    json_path = os.path.join(seq_path, "visible.json")
    
    if not os.path.exists(video_path) or not os.path.exists(json_path):
        return {"status": "error", "message": f"Missing files in {seq_path}"}
        
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        exist = data.get("exist", [])
        bboxes = data.get("gt_rect", [])
        
        absent = [1 if e == 0 else 0 for e in exist]
    except Exception as e:
        return {"status": "error", "message": f"Error reading json in {seq_path}: {e}"}

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
        before_start = max(0, start_idx - num_before_frames)
        before_end = start_idx
        
        after_start = end_idx
        after_end = min(total_frames, end_idx + num_after_frames)
        
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
        
        for frame_idx in needed_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            if frame_idx >= len(bboxes):
                continue
                
            bbox = bboxes[frame_idx]
            # [x, y, w, h] format
            if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
                continue
                
            crop = crop_and_resize(frame, bbox, bbox_padding, crop_size)
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
                "language_description": "",
                "attributes": []
            })
            
    cap.release()
    
    return {
        "status": "success",
        "seq_name": seq_name,
        "pairs": pairs
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgbt-data-dir", default="../Anti-UAV-RGBT")
    parser.add_argument("--processed-dir", default="../UAVAntiUAV/processed")
    args = parser.parse_args()
    
    rgbt_data_dir = args.rgbt_data_dir
    processed_dir = args.processed_dir
    
    pairs_train_json = os.path.join(processed_dir, "pairs_train.json")
    pairs_test_json = os.path.join(processed_dir, "pairs_test.json")
    
    print(f"Loading existing JSON files...")
    with open(pairs_train_json, "r") as f:
        existing_train_pairs = json.load(f)
    with open(pairs_test_json, "r") as f:
        existing_test_pairs = json.load(f)
        
    all_existing_pids = [p["identity_id"] for p in existing_train_pairs + existing_test_pairs if p["identity_id"] is not None]
    max_id = max(all_existing_pids) if all_existing_pids else -1
    print(f"Current maximum identity_id: {max_id}")
    
    splits = {
        "train": ["train", "val"],
        "test": ["test"]
    }
    
    new_train_pairs = []
    new_test_pairs = []
    
    for out_split, rgbt_splits in splits.items():
        tasks = []
        for rgbt_split in rgbt_splits:
            split_dir = os.path.join(rgbt_data_dir, rgbt_split)
            if not os.path.exists(split_dir):
                continue
                
            seqs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
            for seq_name in seqs:
                seq_path = os.path.join(split_dir, seq_name)
                tasks.append((seq_path, processed_dir, out_split))
                
        out_list = new_train_pairs if out_split == "train" else new_test_pairs
        
        print(f"Start processing {len(tasks)} sequences for {out_split}...")
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_sequence, *task): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Processing {out_split}"):
                res = future.result()
                if res["status"] == "success" and res["pairs"]:
                    out_list.extend(res["pairs"])
                    
    # Assign unique IDs
    unique_new_seqs = set()
    for p in new_train_pairs + new_test_pairs:
        unique_new_seqs.add(p["sequence_id"])
        
    seq_to_id = {seq: max_id + 1 + i for i, seq in enumerate(sorted(list(unique_new_seqs)))}
    
    for p in new_train_pairs:
        p["identity_id"] = seq_to_id[p["sequence_id"]]
        existing_train_pairs.append(p)
        
    for p in new_test_pairs:
        p["identity_id"] = seq_to_id[p["sequence_id"]]
        existing_test_pairs.append(p)
        
    print("Writing updated JSON files...")
    with open(pairs_train_json, "w") as f:
        json.dump(existing_train_pairs, f, indent=4)
        
    with open(pairs_test_json, "w") as f:
        json.dump(existing_test_pairs, f, indent=4)
        
    print(f"DONE! Added {len(new_train_pairs)} pairs to train and {len(new_test_pairs)} pairs to test.")

if __name__ == "__main__":
    main()
