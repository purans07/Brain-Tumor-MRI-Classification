#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/outputs"
FIG_DIR="$OUT_DIR/figures"
TMP_DIR="$(mktemp -d /private/tmp/brain-tumor-demo.XXXXXX)"
FONT="/System/Library/Fonts/Supplemental/Arial.ttf"
OUT="$OUT_DIR/brain_tumor_classifier_demo.mp4"

mkdir -p "$OUT_DIR"

ffmpeg -y -f lavfi -i "color=c=0x0F172A:s=1280x720:d=3:r=30" \
  -vf "drawtext=fontfile=$FONT:text='Brain Tumor MRI Image Classification':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=270,drawtext=fontfile=$FONT:text='TensorFlow Keras and Streamlit prototype':fontcolor=0xCBD5E1:fontsize=28:x=(w-text_w)/2:y=345" \
  -c:v libx264 -pix_fmt yuv420p "$TMP_DIR/01_title.mp4"

ffmpeg -y -loop 1 -i "$FIG_DIR/01_class_distribution.png" -t 4 -r 30 \
  -vf "scale=1100:620:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=white,drawtext=fontfile=$FONT:text=Dataset profile 2443 images across four classes:fontcolor=0x0F172A:fontsize=30:x=40:y=30" \
  -c:v libx264 -pix_fmt yuv420p "$TMP_DIR/02_data.mp4"

ffmpeg -y -loop 1 -i "$FIG_DIR/model_comparison_metrics.png" -t 5 -r 30 \
  -vf "scale=1100:620:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=white,drawtext=fontfile=$FONT:text=Best test result EfficientNetB0 fine tuned:fontcolor=0x0F172A:fontsize=30:x=40:y=30,drawtext=fontfile=$FONT:text=69.11% accuracy and 0.677 macro F1:fontcolor=0x0F766E:fontsize=26:x=40:y=665" \
  -c:v libx264 -pix_fmt yuv420p "$TMP_DIR/03_model.mp4"

ffmpeg -y -loop 1 -i "$OUT_DIR/gradcam/custom_cnn_gradcam_example.png" -t 5 -r 30 \
  -vf "scale=1100:620:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=white,drawtext=fontfile=$FONT:text=Explainability qualitative Grad CAM example:fontcolor=0x0F172A:fontsize=30:x=40:y=30,drawtext=fontfile=$FONT:text=Attention maps require clinical validation:fontcolor=0xB45309:fontsize=24:x=40:y=665" \
  -c:v libx264 -pix_fmt yuv420p "$TMP_DIR/04_gradcam.mp4"

ffmpeg -y -f lavfi -i "color=c=white:s=1280x720:d=4:r=30" \
  -vf "drawtext=fontfile=$FONT:text='Streamlit deployment':fontcolor=0x0F172A:fontsize=48:x=(w-text_w)/2:y=220,drawtext=fontfile=$FONT:text='Upload JPG PNG or JPEG MRI images':fontcolor=0x334155:fontsize=30:x=(w-text_w)/2:y=305,drawtext=fontfile=$FONT:text='Displays class prediction confidence and probabilities':fontcolor=0x334155:fontsize=30:x=(w-text_w)/2:y=360,drawtext=fontfile=$FONT:text='Educational prototype not a medical diagnosis':fontcolor=0xB91C1C:fontsize=26:x=(w-text_w)/2:y=470" \
  -c:v libx264 -pix_fmt yuv420p "$TMP_DIR/05_deployment.mp4"

printf "file '%s'\n" "$TMP_DIR/01_title.mp4" "$TMP_DIR/02_data.mp4" "$TMP_DIR/03_model.mp4" "$TMP_DIR/04_gradcam.mp4" "$TMP_DIR/05_deployment.mp4" > "$TMP_DIR/concat.txt"
ffmpeg -y -f concat -safe 0 -i "$TMP_DIR/concat.txt" -c copy -movflags +faststart "$OUT"

echo "$OUT"
