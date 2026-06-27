#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export OPENAI_MODEL="gemini-3-flash-preview"
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_API_BASE="YOUR_URL"
CKPT=gemini3
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}.json \
  --max_pixels 8294400

export OPENAI_API_KEY="YOUR_API_KEY"
CKPT=seed1_5vl
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}.json \
  --max_pixels 8294400

CKPT=qwen2_5vl
SIZE=7B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=qwen3vl
SIZE=2B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=qwen3vl
SIZE=8B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400


CKPT=qwen3vl
SIZE=32B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

  
CKPT=qwen3vl
SIZE=30B-A3B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=internvl
SIZE=8B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=internvl
SIZE=30B-A3B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=minicpmv
SIZE=8B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=osatlas-7b
SIZE=7B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=uground
SIZE=7B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=cogagent24
SIZE=9B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=opencua
SIZE=7B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=opencua
SIZE=32B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=opencua
SIZE=72B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=uitars
SIZE=7B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=tianxi_7b
SIZE=7B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=holo1_5
SIZE=7B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400  

CKPT=holo1_5
SIZE=72B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=holo2
SIZE=4B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400  

CKPT=holo2
SIZE=8B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=holo2
SIZE=30B-A3B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=ui_venus1_5
SIZE=8B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=ui_venus1_5
SIZE=30B-A3B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400

CKPT=mai_ui
SIZE=8B
python eval_egoxr_gui.py \
  --model_type ${CKPT} \
  --model_name_or_path path/to/your/model \
  --screenspot_imgs path/to/your/inages \
  --screenspot_test path/to/your/annotation \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./results/${CKPT}_${SIZE}.json \
  --max_pixels 8294400