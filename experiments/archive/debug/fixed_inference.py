
def fixed_inference(backbone, transformer, image_tensor, confidence_threshold=0.3):
    """修复后的推理函数"""
    # 设置为评估模式
    backbone.eval()
    transformer.eval()
    
    # 推理
    with jt.no_grad():
        feats = backbone(image_tensor)
        outputs = transformer(feats)
    
    # 后处理 - 使用正确的Jittor API
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
    
    # 获取预测结果 - 修复jt.max的使用
    pred_scores = jt.nn.softmax(pred_logits, dim=-1)
    
    # 正确的Jittor max用法
    max_scores = jt.max(pred_scores[:, :-1], dim=-1)[0]  # 最大值
    pred_classes = jt.argmax(pred_scores[:, :-1], dim=-1)  # 最大值索引
    
    # 过滤预测
    high_conf_mask = max_scores > confidence_threshold
    
    if high_conf_mask.sum() > 0:
        high_conf_boxes = pred_boxes[high_conf_mask]
        high_conf_classes = pred_classes[high_conf_mask]
        high_conf_scores = max_scores[high_conf_mask]
        
        return high_conf_boxes, high_conf_classes, high_conf_scores
    else:
        return None, None, None
