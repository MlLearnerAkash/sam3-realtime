WORKDIR=/data/opervu/ws/sam3/sam3-realtime
eval "$(conda shell.bash hook)"
conda activate /home/opervu-user/miniconda3/envs/lang2segtrack
cd "$WORKDIR" && PYTHONPATH=.:$PYTHONPATH python scripts/inference/video_stream.py --stream_type video --video_path /data/dataset/demo_video/output.mp4 --counting_region 0.35 0.35 0.85 0.85