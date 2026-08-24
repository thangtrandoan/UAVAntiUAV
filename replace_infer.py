import os
import re

with open("infer.py", "r") as f:
    content = f.read()

new_main = """
def run_sequence(seq_dir, model, device, transform, cfg, inf_cfg, out_base=None):
    seq_name = os.path.basename(os.path.normpath(seq_dir))
    video_path = os.path.join(seq_dir, f"{seq_name}.mp4")
    gt_path = os.path.join(seq_dir, "groundtruth_rect.txt")
    absent_path = os.path.join(seq_dir, "absent.txt")
    
    if out_base:
        out_dir = os.path.join(out_base, seq_name)
    else:
        out_dir = inf_cfg.get('out_dir', f"infer_output/{seq_name}")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    metrics_path = os.path.join(out_dir, "metrics.txt")
    metrics_file = open(metrics_path, "w")
    _orig_print = builtins.print
    def custom_print(*args_p, **kwargs_p):
        msg = " ".join(str(a) for a in args_p)
        _orig_print(msg, **kwargs_p)
        if not metrics_file.closed:
            metrics_file.write(msg + "\\n")
            metrics_file.flush()
    builtins.print = custom_print

    output_video_name = inf_cfg.get('output_video', 'output_reid.mp4')
    final_output_path = os.path.join(out_dir, os.path.basename(output_video_name))
    
    bboxes = []
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 4:
                    bboxes.append([int(float(p)) for p in parts[:4]])
                else:
                    bboxes.append([0, 0, 0, 0])
                    
    absent = []
    if os.path.exists(absent_path):
        with open(absent_path, "r") as f:
            absent = [int(line.strip()) for line in f if line.strip().isdigit()]

    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        metrics_file.close()
        builtins.print = _orig_print
        return None

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(final_output_path, fourcc, fps_video, (width, height))
    
    pipeline = SeqReIDPipeline(model, device, inf_cfg)
    
    frame_idx = 0
    print(f"Starting OOP Sequence Inference Stream for {seq_name}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        is_absent = absent[frame_idx] == 1 if frame_idx < len(absent) else True
        bbox = bboxes[frame_idx] if frame_idx < len(bboxes) else [0,0,0,0]
        
        display_frame = frame.copy()
        
        pipeline.process_frame(frame, bbox, is_absent, frame_idx, transform)
        pipeline.draw_ui(display_frame, bbox, frame_idx)
        
        out_vid.write(display_frame)
        frame_idx += 1

    cap.release()
    out_vid.release()
    print("Inference completed!")
    
    # Generate Performance Metrics
    metrics_report = ["\\n--- PERFORMANCE METRICS ---"]
    avg_cnn = 0.0
    avg_mamba = 0.0
    throughput = 0.0
    
    if pipeline.metrics_cnn_times:
        avg_cnn = np.mean(pipeline.metrics_cnn_times)
        throughput = 1000.0 / avg_cnn if avg_cnn > 0 else 0.0
        metrics_report.append(f"Avg CNN Feature Extraction : {avg_cnn:.2f} ms")
        metrics_report.append(f"Avg System Throughput      : {throughput:.2f} FPS")
    if pipeline.metrics_mamba_times:
        avg_mamba = np.mean(pipeline.metrics_mamba_times)
        metrics_report.append(f"Avg Mamba + Head Time      : {avg_mamba:.2f} ms")
    metrics_report.append(f"Re-acquisition Latency     : {pipeline.reid_latency_frames} frames" if pipeline.reid_latency_frames >= 0 else "Re-acquisition Latency     : N/A")
    metrics_report.append(f"False Alarms (Fine Fails)  : {pipeline.false_alarms}")
    
    print("\\n".join(metrics_report))
    metrics_file.close()
    builtins.print = _orig_print
    
    return avg_cnn, avg_mamba, throughput, pipeline.reid_latency_frames, pipeline.false_alarms

def main():
    args = parse_args()
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            
    inf_cfg = cfg.get('infer', {})
    seq_dir_arg = args.seq_dir or inf_cfg.get('seq_dir')
    if not seq_dir_arg:
        print("Error: --seq-dir must be provided.")
        return

    print(f"Initializing Mamba ReID Model...")
    model = UAVReIDNet()
    model_path = inf_cfg.get('model_path', './best_model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {}
        model_state = model.state_dict()
        for k, v in state_dict.items():
            new_k = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
            if new_k in model_state and v.shape != model_state[new_k].shape:
                continue
            new_state_dict[new_k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Checkpoint {model_path} not found. Running with random weights.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if seq_dir_arg.lower() == "all":
        base_test_dir = "./data/UAV-Anti-UAV/Test"
        all_dirs = [os.path.join(base_test_dir, d) for d in sorted(os.listdir(base_test_dir)) if os.path.isdir(os.path.join(base_test_dir, d))]
        valid_seqs = []
        for d in all_dirs:
            absent_path = os.path.join(d, "absent.txt")
            if os.path.exists(absent_path):
                with open(absent_path, "r") as f:
                    absent = [int(line.strip()) for line in f if line.strip().isdigit()]
                if 1 in absent:
                    valid_seqs.append(d)
        print(f"Found {len(valid_seqs)} sequences with disappearance events.")
        
        all_cnn = []
        all_mamba = []
        all_throughput = []
        all_latency = []
        all_false_alarms = []
        
        for sdir in valid_seqs:
            res = run_sequence(sdir, model, device, transform, cfg, inf_cfg, out_base="./infer_pipeline_1")
            if res:
                c, m, t, l, f = res
                all_cnn.append(c)
                all_mamba.append(m)
                all_throughput.append(t)
                if l >= 0:
                    all_latency.append(l)
                all_false_alarms.append(f)
                
        # Calculate averages
        avg_cnn = np.mean(all_cnn) if all_cnn else 0.0
        avg_mamba = np.mean(all_mamba) if all_mamba else 0.0
        avg_throughput = np.mean(all_throughput) if all_throughput else 0.0
        avg_latency = np.mean(all_latency) if all_latency else 0.0
        sum_false_alarms = int(np.sum(all_false_alarms)) if all_false_alarms else 0
        
        print("\\n=== AGGREGATED METRICS ===")
        print(f"Avg CNN Feature Extraction : {avg_cnn:.2f} ms")
        print(f"Avg System Throughput      : {avg_throughput:.2f} FPS")
        print(f"Avg Mamba + Head Time      : {avg_mamba:.2f} ms")
        print(f"Avg Re-acquisition Latency : {avg_latency:.2f} frames")
        print(f"Total False Alarms         : {sum_false_alarms}")
        
        # Save to summary text file
        os.makedirs("./infer_pipeline_1", exist_ok=True)
        with open("./infer_pipeline_1/summary_metrics.txt", "w") as sf:
            sf.write("=== AGGREGATED METRICS ===\\n")
            sf.write(f"Avg CNN Feature Extraction : {avg_cnn:.2f} ms\\n")
            sf.write(f"Avg System Throughput      : {avg_throughput:.2f} FPS\\n")
            sf.write(f"Avg Mamba + Head Time      : {avg_mamba:.2f} ms\\n")
            sf.write(f"Avg Re-acquisition Latency : {avg_latency:.2f} frames\\n")
            sf.write(f"Total False Alarms         : {sum_false_alarms}\\n")
            
    else:
        run_sequence(seq_dir_arg, model, device, transform, cfg, inf_cfg, out_base=None)

if __name__ == "__main__":
    main()
"""

# Find the start of def main(): and replace everything after it
idx = content.find("def main():")
if idx != -1:
    new_content = content[:idx] + new_main
    with open("infer.py", "w") as f:
        f.write(new_content)
    print("Successfully replaced main and added run_sequence.")
else:
    print("Could not find def main():")

