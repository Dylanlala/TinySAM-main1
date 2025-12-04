#!/bin/bash
# 快速推理脚本

YOLO_WEIGHTS="runs/detect/yolo11n_kvasir/weights/best.pt"
SAM_WEIGHTS="results_tinysam_kvasir/best_model.pth"
TEST_IMAGES="Kvasir-SEG/test/images"
TEST_MASKS="Kvasir-SEG/masks"
OUTPUT_DIR="inference_results_kvasir"

echo "开始推理..."
python inference_yolo_tinysam.py \
    --yolo-weights $YOLO_WEIGHTS \
    --sam-weights $SAM_WEIGHTS \
    --test-images $TEST_IMAGES \
    --output-dir $OUTPUT_DIR \
    --yolo-conf 0.25 \
    --yolo-iou 0.45

echo ""
echo "开始评估..."
python evaluate_yolo_tinysam.py \
    --yolo-weights $YOLO_WEIGHTS \
    --sam-weights $SAM_WEIGHTS \
    --test-images $TEST_IMAGES \
    --test-masks $TEST_MASKS \
    --output-dir evaluation_results_kvasir \
    --yolo-conf 0.25 \
    --yolo-iou 0.45

echo ""
echo "分析失败案例..."
python analyze_failure_cases.py \
    --evaluation-report evaluation_results_kvasir/evaluation_report.json \
    --inference-results $OUTPUT_DIR

echo ""
echo "完成！结果保存在："
echo "  - 推理结果: $OUTPUT_DIR"
echo "  - 评估报告: evaluation_results_kvasir/"
echo "  - 失败分析: failure_analysis/"

