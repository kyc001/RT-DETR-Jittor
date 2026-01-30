"""RT-DETR Jittor 模块测试脚本"""

import sys
sys.path.insert(0, '/wanyuhao/keyunchao/project/RT-DETR-Jittor-main/RT-DETR-Jittor-main/rtdetr_jittor')

import jittor as jt
jt.flags.use_cuda = 0  # CPU测试

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters())

def test_backbone():
    """测试ResNet Backbone"""
    print("=" * 50)
    print("测试 ResNet Backbone")
    print("=" * 50)

    from src.nn.backbone.resnet import PResNet

    # 测试ResNet50
    backbone = PResNet(depth=50, variant='d', return_idx=[1, 2, 3])
    x = jt.randn(1, 3, 640, 640)

    outs = backbone(x)

    print(f"输入形状: {x.shape}")
    print(f"输出数量: {len(outs)}")
    for i, out in enumerate(outs):
        print(f"  输出 {i}: {out.shape}")

    params = count_parameters(backbone)
    print(f"参数量: {params:,} ({params/1e6:.2f}M)")
    print(f"输出通道: {backbone.out_channels}")
    print(f"输出步幅: {backbone.out_strides}")
    print("✓ Backbone 测试通过\n")

    return backbone

def test_encoder():
    """测试HybridEncoder"""
    print("=" * 50)
    print("测试 HybridEncoder")
    print("=" * 50)

    from src.zoo.rtdetr.hybrid_encoder import HybridEncoder

    encoder = HybridEncoder(
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        use_encoder_idx=[2],
        num_encoder_layers=1,
        expansion=1.0,
        depth_mult=1.0,
        act='silu'
    )

    # 模拟backbone输出
    feats = [
        jt.randn(1, 512, 80, 80),
        jt.randn(1, 1024, 40, 40),
        jt.randn(1, 2048, 20, 20)
    ]

    outs = encoder(feats)

    print(f"输入特征数量: {len(feats)}")
    for i, feat in enumerate(feats):
        print(f"  输入 {i}: {feat.shape}")

    print(f"输出特征数量: {len(outs)}")
    for i, out in enumerate(outs):
        print(f"  输出 {i}: {out.shape}")

    params = count_parameters(encoder)
    print(f"参数量: {params:,} ({params/1e6:.2f}M)")
    print("✓ Encoder 测试通过\n")

    return encoder

def test_decoder():
    """测试RTDETRTransformer"""
    print("=" * 50)
    print("测试 RTDETRTransformer (Decoder)")
    print("=" * 50)

    from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer

    decoder = RTDETRTransformer(
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        aux_loss=True
    )

    # 模拟encoder输出
    feats = [
        jt.randn(1, 256, 80, 80),
        jt.randn(1, 256, 40, 40),
        jt.randn(1, 256, 20, 20)
    ]

    decoder.eval()  # 测试模式
    outputs = decoder(feats)

    print(f"输入特征数量: {len(feats)}")
    for i, feat in enumerate(feats):
        print(f"  输入 {i}: {feat.shape}")

    print(f"输出键: {outputs.keys()}")
    print(f"  pred_logits: {outputs['pred_logits'].shape}")
    print(f"  pred_boxes: {outputs['pred_boxes'].shape}")

    params = count_parameters(decoder)
    print(f"参数量: {params:,} ({params/1e6:.2f}M)")
    print("✓ Decoder 测试通过\n")

    return decoder

def test_criterion():
    """测试Criterion"""
    print("=" * 50)
    print("测试 Criterion")
    print("=" * 50)

    from src.zoo.rtdetr.rtdetr_criterion import build_criterion

    criterion = build_criterion(num_classes=80)

    # 模拟模型输出
    outputs = {
        'pred_logits': jt.randn(2, 300, 80),
        'pred_boxes': jt.sigmoid(jt.randn(2, 300, 4))
    }

    # 模拟目标
    targets = [
        {'labels': jt.array([0, 1, 2]), 'boxes': jt.rand(3, 4)},
        {'labels': jt.array([5, 10]), 'boxes': jt.rand(2, 4)}
    ]

    losses = criterion(outputs, targets)

    print(f"损失键: {losses.keys()}")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    params = count_parameters(criterion)
    print(f"参数量: {params:,}")
    print("✓ Criterion 测试通过\n")

    return criterion

def test_full_model():
    """测试完整模型"""
    print("=" * 50)
    print("测试完整模型 (Backbone + Encoder + Decoder)")
    print("=" * 50)

    from src.nn.backbone.resnet import PResNet
    from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
    from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer

    # Backbone
    backbone = PResNet(depth=50, variant='d', return_idx=[1, 2, 3])

    # Encoder
    encoder = HybridEncoder(
        in_channels=backbone.out_channels,
        feat_strides=backbone.out_strides,
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        use_encoder_idx=[2],
        num_encoder_layers=1,
    )

    # Decoder
    decoder = RTDETRTransformer(
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=encoder.out_channels,
        feat_strides=encoder.out_strides,
        num_levels=3,
    )

    # 测试
    x = jt.randn(1, 3, 640, 640)

    # Forward
    backbone_outs = backbone(x)
    encoder_outs = encoder(backbone_outs)

    decoder.eval()
    outputs = decoder(encoder_outs)

    print(f"输入: {x.shape}")
    print(f"Backbone输出: {[o.shape for o in backbone_outs]}")
    print(f"Encoder输出: {[o.shape for o in encoder_outs]}")
    print(f"Decoder输出:")
    print(f"  pred_logits: {outputs['pred_logits'].shape}")
    print(f"  pred_boxes: {outputs['pred_boxes'].shape}")

    total_params = count_parameters(backbone) + count_parameters(encoder) + count_parameters(decoder)
    print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Backbone: {count_parameters(backbone):,}")
    print(f"  Encoder: {count_parameters(encoder):,}")
    print(f"  Decoder: {count_parameters(decoder):,}")
    print("✓ 完整模型测试通过\n")

def main():
    print("\n" + "=" * 60)
    print("RT-DETR Jittor 模块测试")
    print("=" * 60 + "\n")

    try:
        test_backbone()
    except Exception as e:
        print(f"✗ Backbone 测试失败: {e}\n")

    try:
        test_encoder()
    except Exception as e:
        print(f"✗ Encoder 测试失败: {e}\n")

    try:
        test_decoder()
    except Exception as e:
        print(f"✗ Decoder 测试失败: {e}\n")

    try:
        test_criterion()
    except Exception as e:
        print(f"✗ Criterion 测试失败: {e}\n")

    try:
        test_full_model()
    except Exception as e:
        print(f"✗ 完整模型测试失败: {e}\n")

    print("=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == '__main__':
    main()
