import argparse
import os
import torch
import yaml
from model import UAVReIDNet, load_checkpoint_verbose

def export_split(model, args, visual_dim):
    print("Exporting Backbone...")
    class BackboneModel(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            
        def forward(self, x):
            feats = self.backbone(x)
            if isinstance(feats, tuple):
                if isinstance(feats[0], tuple):
                    return torch.cat([feats[0][0], feats[0][1]], dim=-1)
                else:
                    return torch.cat([feats[0], feats[1]], dim=-1)
            return feats

    backbone_model = BackboneModel(model.backbone)
    backbone_model.eval()
    dummy_img = torch.randn(args.batch_size, 3, args.img_size, args.img_size).cuda()
    
    torch.onnx.export(
        backbone_model,
        dummy_img,
        args.prefix + "backbone.onnx",
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input_img'],
        output_names=['cnn_feat'],
        dynamic_axes={'input_img': {0: 'batch_size'}, 'cnn_feat': {0: 'batch_size'}}
    )
    print("Saved uav_reid_backbone.onnx")

    print("Exporting Temporal Head...")
    class TemporalModel(torch.nn.Module):
        def __init__(self, temporal_encoder, head):
            super().__init__()
            self.temporal_encoder = temporal_encoder
            self.head = head
            
        def forward(self, seq_feats):
            visual_feat = seq_feats.mean(dim=1)
            temporal_token, _ = self.temporal_encoder(seq_feats)
            bn_feat = self.head(visual_feat, temporal_token)
            return torch.nn.functional.normalize(bn_feat, p=2, dim=1)

    temporal_model = TemporalModel(model.temporal_encoder, model.head)
    temporal_model.eval()
    dummy_seq = torch.randn(args.batch_size, args.num_frames, visual_dim).cuda()
    
    torch.onnx.export(
        temporal_model,
        dummy_seq,
        args.prefix + "temporal.onnx",
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['seq_feats'],
        output_names=['reid_embed'],
        dynamic_axes={'seq_feats': {0: 'batch_size', 1: 'num_frames'}, 'reid_embed': {0: 'batch_size'}}
    )
    print("Saved uav_reid_temporal.onnx")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_jetson.yaml', type=str)
    parser.add_argument('--checkpoint', default='./best_model.pth', type=str)
    parser.add_argument('--mode', choices=['split', 'unified', 'both'], default='split')
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--num-frames', default=16, type=int)
    parser.add_argument('--img-size', default=224, type=int)
    parser.add_argument('--opset', default=17, type=int)
    parser.add_argument('--temporal-type', default='mamba', type=str, choices=['mamba', 'attention'])
    parser.add_argument('--temporal-pool', default='mean', type=str, choices=['mean', 'attn'])
    parser.add_argument('--backbone', default=None, type=str, help="Override backbone from config")
    parser.add_argument('--econvnext', action='store_true', help="Use econvnext backbone")
    parser.add_argument('--prefix', default='uav_reid_', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get('train', {})
    backbone = args.backbone if args.backbone else train_cfg.get('backbone', 'dinov3_convnext')
    
    print(f"Loading UAVReIDNet (Backbone: {backbone}, Temporal: {args.temporal_type}, Pool: {args.temporal_pool})...")
    model = UAVReIDNet(freeze_backbone=False, backbone=backbone, temporal_type=args.temporal_type, temporal_pool=args.temporal_pool, econvnext=args.econvnext)

    if os.path.exists(args.checkpoint):
        load_checkpoint_verbose(model, args.checkpoint, tag="export")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Exporting random weights.")

    model.eval()
    model.cuda()

    visual_dim = 960 if backbone == "dinov3_convnext" else 2560
    
    if args.mode in ['split', 'both']:
        export_split(model, args, visual_dim)
        
    print("Done!")
