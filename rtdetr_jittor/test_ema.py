#!/usr/bin/env python3
"""
测试EMA模块和权重转换功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jittor as jt
jt.flags.use_cuda = 0  # 使用CPU测试

print("=" * 60)
print("RT-DETR Jittor EMA模块测试")
print("=" * 60)


def test_ema():
    """测试EMA模块"""
    print("\n[1] 测试EMA模块导入")

    try:
        from src.optim.ema import ModelEMA, create_ema
        print("    EMA模块导入成功")
    except Exception as e:
        print(f"    EMA模块导入失败: {e}")
        return False

    print("\n[2] 创建测试模型")
    import jittor.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 8 * 8, 10)

        def execute(self, x):
            x = nn.relu(self.bn1(self.conv1(x)))
            x = nn.max_pool2d(x, 2)
            x = nn.relu(self.conv2(x))
            x = nn.max_pool2d(x, 2)
            x = x.view(x.shape[0], -1)
            return self.fc(x)

    model = SimpleModel()
    print(f"    模型创建成功，参数量: {sum(p.numel() for p in model.parameters())}")

    print("\n[3] 创建EMA")
    ema = ModelEMA(model, decay=0.9999, warmups=100)
    print(f"    EMA创建成功: {ema}")

    print("\n[4] 模拟训练循环并测试EMA更新")
    optimizer = jt.optim.Adam(model.parameters(), lr=0.001)

    for i in range(5):
        # 模拟前向传播
        x = jt.randn(2, 3, 32, 32)
        y = model(x)
        loss = y.sum()

        # 反向传播
        optimizer.step(loss)

        # EMA更新
        ema.update(model)

        print(f"    Step {i+1}: EMA updates = {ema.updates}, decay = {ema.decay_fn(ema.updates):.6f}")

    print("\n[5] 测试EMA state_dict 保存/加载")
    state = ema.state_dict()
    print(f"    state_dict keys: {state.keys()}")

    # 创建新的EMA并加载状态
    new_ema = ModelEMA(model, decay=0.9999, warmups=100)
    new_ema.load_state_dict(state)
    print(f"    加载后 updates = {new_ema.updates}")

    print("\n[6] 比较模型和EMA模型参数")
    model_params = list(model.parameters())
    ema_params = list(ema.module.parameters())

    for i, (mp, ep) in enumerate(zip(model_params[:2], ema_params[:2])):
        diff = (mp - ep).abs().mean().item()
        print(f"    参数{i} 差异: {diff:.6f}")

    print("\n✓ EMA模块测试通过")
    return True


def test_trainer_with_ema():
    """测试带EMA的训练器"""
    print("\n" + "=" * 60)
    print("测试带EMA的训练器")
    print("=" * 60)

    try:
        from src.components.trainer import RTDETRTrainer, create_trainer
        print("\n[1] 训练器模块导入成功")
    except Exception as e:
        print(f"\n[1] 训练器模块导入失败: {e}")
        return False

    import jittor.nn as nn

    # 创建简单模型
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def execute(self, x, targets=None):
            return {'pred_logits': self.fc(x), 'pred_boxes': jt.zeros(x.shape[0], 5, 4)}

    class DummyCriterion(nn.Module):
        def execute(self, outputs, targets):
            return {'loss': outputs['pred_logits'].sum()}

    model = DummyModel()
    criterion = DummyCriterion()
    optimizer = jt.optim.Adam(model.parameters(), lr=0.001)

    print("\n[2] 创建带EMA的训练器")
    trainer = RTDETRTrainer(
        model, criterion, optimizer,
        save_dir="./test_results",
        use_ema=True, ema_decay=0.9999, ema_warmups=100
    )

    if trainer.ema is not None:
        print("    训练器EMA已启用")
    else:
        print("    警告: EMA未启用")

    print("\n[3] 测试评估模型获取")
    eval_model = trainer.get_eval_model()
    if trainer.ema is not None:
        assert eval_model is trainer.ema.module, "应该返回EMA模型"
        print("    正确返回EMA模型用于评估")
    else:
        assert eval_model is trainer.model, "应该返回原始模型"
        print("    正确返回原始模型用于评估")

    print("\n✓ 带EMA的训练器测试通过")
    return True


def test_weight_conversion():
    """测试权重转换工具"""
    print("\n" + "=" * 60)
    print("测试权重转换工具")
    print("=" * 60)

    try:
        from tools.convert_weights import convert_pytorch_to_jittor, convert_jittor_to_pytorch
        print("\n[1] 权重转换工具导入成功")
    except Exception as e:
        print(f"\n[1] 权重转换工具导入失败: {e}")
        return False

    print("\n✓ 权重转换工具导入测试通过")
    print("  (完整测试需要PyTorch和实际权重文件)")
    return True


def main():
    results = []

    # 测试EMA模块
    results.append(("EMA模块", test_ema()))

    # 测试带EMA的训练器
    results.append(("带EMA训练器", test_trainer_with_ema()))

    # 测试权重转换
    results.append(("权重转换工具", test_weight_conversion()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n所有测试通过!")
    else:
        print("\n部分测试失败，请检查错误信息")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
