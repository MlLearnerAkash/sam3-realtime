WORKDIR=/home/opervu/ws/sam3-realtime

# Set USE_HDR=1 to use the Arena HDR camera; 0 to read a video file.
USE_HDR=1

VIDEO_PATH="$WORKDIR/assets/videos/bedroom.mp4"
COUNTING_REGION="0.35 0.35 0.85 0.85"

if [ "$USE_HDR" = "1" ]; then
    STREAM_ARGS="--stream_type hdr"
else
    STREAM_ARGS="--stream_type video --video_path $VIDEO_PATH"
fi

cd "$WORKDIR" && pixi run python scripts/inference/video_stream.py $STREAM_ARGS --counting_region $COUNTING_REGION