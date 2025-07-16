
def fixed_inference_with_correct_api(backbone, transformer, image_tensor, confidence_threshold=0.3):
    """使用正确Jittor API的推理函数"""
    # 设置为评估模式
    backbone.eval()
    transformer.eval()
    
    # 推理
    with jt.no_grad():
        feats = backbone(image_tensor)
        outputs = transformer(feats)
    
    # 后处理
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
    
    # 获取预测结果 - 使用正确的Jittor API
    pred_scores = jt.nn.softmax(pred_logits, dim=-1)
    
    # 方法1: 尝试使用jt.max返回元组
    try:
        max_result = jt.max(pred_scores[:, :-1], dim=-1)
        if isinstance(max_result, tuple):
            max_scores, pred_classes = max_result
        else:
            max_scores = max_result
            pred_classes = jt.argmax(pred_scores[:, :-1], dim=-1)
    except:
        # 方法2: 分别使用max和argmax
        max_scores = jt.max(pred_scores[:, :-1], dim=-1, keepdims=False)
        pred_classes = jt.argmax(pred_scores[:, :-1], dim=-1)
    
    # 过滤预测
    high_conf_mask = max_scores > confidence_threshold
    
    if high_conf_mask.sum() > 0:
        high_conf_boxes = pred_boxes[high_conf_mask]
        high_conf_classes = pred_classes[high_conf_mask]
        high_conf_scores = max_scores[high_conf_mask]
        
        return high_conf_boxes, high_conf_classes, high_conf_scores
    else:
        return None, None, None
