import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
from tqdm import tqdm
import yaml

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

def process_sequence(anno_name, uav123_data_dir, output_base, split, num_before_frames=16, num_after_frames=16, frame_stride=1, bbox_padding=0.2, crop_size=256):
    anno_dir = os.path.join(uav123_data_dir, "anno", "UAV123")
    seq_dir_base = os.path.join(uav123_data_dir, "data_seq", "UAV123")
    
    anno_path = os.path.join(anno_dir, f"{anno_name}.txt")
    att_path = os.path.join(anno_dir, "att", f"{anno_name}.txt")
    
    # Determine image folder
    img_folder = os.path.join(seq_dir_base, anno_name)
    if not os.path.exists(img_folder):
        base_name = anno_name.rsplit('_', 1)[0]
        img_folder = os.path.join(seq_dir_base, base_name)
        
    if not os.path.exists(img_folder) or not os.path.exists(anno_path):
        return {"status": "error", "message": f"Missing files for {anno_name}"}
        
    # Find exact frame files
    all_imgs = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg')])
    total_frames = len(all_imgs)
    
    if total_frames == 0:
        return {"status": "error", "message": f"No images found in {img_folder}"}
        
    try:
        bboxes = []
        absent = []
        with open(anno_path, "r") as f:
            for line in f:
                if "NaN" in line:
                    absent.append(1)
                    bboxes.append([0, 0, 0, 0])
                else:
                    absent.append(0)
                    bboxes.append([int(float(x)) for x in line.strip().split(',')[:4]])
                    
        attributes = []
        if os.path.exists(att_path):
            with open(att_path, "r") as f:
                parts = f.read().strip().split(',')
                attributes = [int(x) for x in parts if x.strip().isdigit()]
                
    except Exception as e:
        return {"status": "error", "message": f"Error reading annotations for {anno_name}: {e}"}

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
        return {"status": "skipped", "message": f"No disappearance in {anno_name}", "seq_name": anno_name}
        
    pairs = []
    
    for event_idx, (start_idx, end_idx) in enumerate(events):
        t1 = start_idx - 1
        before_sampled = [t1 - i * frame_stride for i in range(num_before_frames)]
        before_sampled = [f for f in before_sampled if f >= 0]
        before_sampled.sort()
        
        t2 = end_idx
        after_sampled = [t2 + i * frame_stride for i in range(num_after_frames)]
        after_sampled = [f for f in after_sampled if f < min(total_frames, len(bboxes))]
        after_sampled.sort()
        
        needed_frames = sorted(list(set(before_sampled + after_sampled)))
        
        if not needed_frames:
            continue
            
        event_out_dir = os.path.join(output_base, split, f"{anno_name}_event_{event_idx}")
        before_dir = os.path.join(event_out_dir, "before")
        after_dir = os.path.join(event_out_dir, "after")
        os.makedirs(before_dir, exist_ok=True)
        os.makedirs(after_dir, exist_ok=True)
        
        before_frames_files = []
        after_frames_files = []
        
        for frame_idx in needed_frames:
            img_path = os.path.join(img_folder, all_imgs[frame_idx])
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            bbox = bboxes[frame_idx]
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
                "sequence_id": anno_name,
                "event_index": event_idx,
                "identity_id": None, 
                "gallery_frames": before_frames_files,
                "query_frames": after_frames_files,
                "gallery_dir": f"{anno_name}_event_{event_idx}/before",
                "query_dir": f"{anno_name}_event_{event_idx}/after",
                "disappearance_duration_frames": end_idx - start_idx,
                "language_description": "",
                "attributes": attributes
            })
            
    return {
        "status": "success",
        "seq_name": anno_name,
        "pairs": pairs
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav123-data-dir", default="./data/UAV123")
    parser.add_argument("--processed-dir", default="./processed")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config (overrides args)")
    
    # Fallback default args if config not provided
    parser.add_argument("--num-before-frames", type=int, default=16)
    parser.add_argument("--num-after-frames", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--bbox-padding", type=float, default=0.2)
    parser.add_argument("--crop-size", type=int, default=256)
    
    args = parser.parse_args()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        if "data_pipeline" in cfg:
            dp = cfg["data_pipeline"]
            args.num_before_frames = dp.get("num_before_frames", args.num_before_frames)
            args.num_after_frames = dp.get("num_after_frames", args.num_after_frames)
            args.frame_stride = dp.get("frame_stride", args.frame_stride)
            args.bbox_padding = dp.get("bbox_padding", args.bbox_padding)
            args.crop_size = dp.get("crop_size", args.crop_size)
    
    uav123_data_dir = args.uav123_data_dir
    processed_dir = args.processed_dir
    
    query_train_json = os.path.join(processed_dir, "query_train.json")
    gallery_train_json = os.path.join(processed_dir, "gallery_train.json")
    query_test_json = os.path.join(processed_dir, "query_test.json")
    gallery_test_json = os.path.join(processed_dir, "gallery_test.json")
    
    print(f"Loading existing JSON files...")
    if os.path.exists(query_train_json):
        with open(query_train_json, "r") as f:
            existing_query_train = json.load(f)
    else:
        existing_query_train = []
        
    if os.path.exists(gallery_train_json):
        with open(gallery_train_json, "r") as f:
            existing_gallery_train = json.load(f)
    else:
        existing_gallery_train = []

    if os.path.exists(query_test_json):
        with open(query_test_json, "r") as f:
            existing_query_test = json.load(f)
    else:
        existing_query_test = []
        
    if os.path.exists(gallery_test_json):
        with open(gallery_test_json, "r") as f:
            existing_gallery_test = json.load(f)
    else:
        existing_gallery_test = []
        
    all_existing_pids = [p["identity_id"] for p in existing_query_train + existing_query_test if p["identity_id"] is not None]
    max_id = max(all_existing_pids) if all_existing_pids else -1
    print(f"Current maximum identity_id: {max_id}")
    
    anno_dir = os.path.join(uav123_data_dir, "anno", "UAV123")
    seqs = sorted([f[:-4] for f in os.listdir(anno_dir) if f.endswith('.txt')])
    
    import hashlib
    tasks = []
    
    for anno_name in seqs:
        # Deterministic split: 80% train, 20% test
        # Group by base name (e.g., group1) so sub-sequences (group1_1, group1_2) fall into the same split
        base_name = anno_name.rsplit('_', 1)[0]
        hash_val = int(hashlib.md5(base_name.encode()).hexdigest(), 16)
        out_split = "test" if hash_val % 5 == 0 else "train"
        
        tasks.append((anno_name, uav123_data_dir, processed_dir, out_split, args.num_before_frames, args.num_after_frames, args.frame_stride, args.bbox_padding, args.crop_size))
            
    new_train_pairs = []
    new_test_pairs = []
    
    print(f"Start processing {len(tasks)} sequences...")
    
    # We map futures to their out_split so we know where to append results
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_split = {executor.submit(process_sequence, *task): task[3] for task in tasks}
        for future in tqdm(as_completed(future_to_split), total=len(tasks), desc="Processing UAV123"):
            out_split = future_to_split[future]
            res = future.result()
            if res["status"] == "success" and res["pairs"]:
                if out_split == "train":
                    new_train_pairs.extend(res["pairs"])
                else:
                    new_test_pairs.extend(res["pairs"])
                
    unique_new_seqs = set()
    for p in new_train_pairs + new_test_pairs:
        unique_new_seqs.add(p["sequence_id"])
        
    seq_to_id = {seq: max_id + 1 + i for i, seq in enumerate(sorted(list(unique_new_seqs)))}
    
    for p in new_train_pairs:
        p["identity_id"] = seq_to_id[p["sequence_id"]]
        existing_gallery_train.append({
            "sequence_id": p["sequence_id"],
            "event_index": p["event_index"],
            "identity_id": p["identity_id"],
            "frames": p["gallery_frames"],
            "frame_dir": p["gallery_dir"]
        })
        existing_query_train.append({
            "sequence_id": p["sequence_id"],
            "event_index": p["event_index"],
            "identity_id": p["identity_id"],
            "frames": p["query_frames"],
            "frame_dir": p["query_dir"],
            "disappearance_duration_frames": p["disappearance_duration_frames"],
            "language_description": p["language_description"],
            "attributes": p["attributes"]
        })
        
    for p in new_test_pairs:
        p["identity_id"] = seq_to_id[p["sequence_id"]]
        existing_gallery_test.append({
            "sequence_id": p["sequence_id"],
            "event_index": p["event_index"],
            "identity_id": p["identity_id"],
            "frames": p["gallery_frames"],
            "frame_dir": p["gallery_dir"]
        })
        existing_query_test.append({
            "sequence_id": p["sequence_id"],
            "event_index": p["event_index"],
            "identity_id": p["identity_id"],
            "frames": p["query_frames"],
            "frame_dir": p["query_dir"],
            "disappearance_duration_frames": p["disappearance_duration_frames"],
            "language_description": p["language_description"],
            "attributes": p["attributes"]
        })
        
    print("Writing updated JSON files...")
    with open(query_train_json, "w") as f:
        json.dump(existing_query_train, f, indent=4)
    with open(gallery_train_json, "w") as f:
        json.dump(existing_gallery_train, f, indent=4)
    with open(query_test_json, "w") as f:
        json.dump(existing_query_test, f, indent=4)
    with open(gallery_test_json, "w") as f:
        json.dump(existing_gallery_test, f, indent=4)
        
    print(f"DONE! Added {len(new_train_pairs)} pairs to UAV123 train and {len(new_test_pairs)} pairs to UAV123 test.")

if __name__ == "__main__":
    main()
