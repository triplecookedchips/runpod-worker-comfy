# Keep CUDA 11.8 as it's most compatible with RunPod
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# AWS environment variables
ENV BUCKET_ENDPOINT_URL=""
ENV BUCKET_ACCESS_KEY_ID=""
ENV BUCKET_SECRET_ACCESS_KEY=""

# System dependencies installation
RUN apt-get update && \
    apt-mark unhold libnccl2 libcudnn8 && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
    python3-dev \
    python3-pip \
    python3.10-venv \
    fonts-dejavu-core \
    rsync \
    git \
    git-lfs \
    jq \
    moreutils \
    aria2 \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libgl1 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    unzip \
    libgoogle-perftools-dev \
    procps \
    build-essential \
    libcublas-11-8 \
    libcublas-dev-11-8 \
    libnccl2 \
    libnccl-dev && \
    apt-mark hold libnccl2 libcudnn8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone ComfyUI and set specific version
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui && \
    cd /comfyui && \
    git reset --hard 20447e9ec92b7e7e3544a6fd2932c31c90333991

WORKDIR /comfyui

# Create all model directories
RUN mkdir -p models/clip_vision \
    models/ipadapter \
    models/checkpoints \
    models/loras \
    models/controlnet \
    models/facerestore_models \
    models/insightface/models \
    models/facedetection \
    models/annotators && \
    chmod -R 755 models

# Install Python dependencies with specific versions for CUDA 11.8
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install --no-cache-dir xformers==0.0.21 \
    && pip3 install -r requirements.txt \
    && pip3 uninstall -y onnxruntime opencv-python opencv-contrib-python opencv-python-headless \
    && pip3 install --no-cache-dir \
        onnxruntime-gpu==1.15.1 \
        runpod \
        requests \
        opencv-python==4.7.0.72 \
        insightface==0.7.3 \
        numpy==1.23.5

# Install custom nodes with specific versions
RUN git clone --depth=1 https://github.com/cubiq/ComfyUI_IPAdapter_plus.git custom_nodes/ComfyUI_IPAdapter_plus && \
    git clone --depth=1 https://github.com/Fannovel16/comfyui_controlnet_aux.git custom_nodes/comfyui_controlnet_aux && \
    git clone --depth=1 https://github.com/cubiq/ComfyUI_essentials.git custom_nodes/ComfyUI_essentials && \
    git clone --depth=1 https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git custom_nodes/ComfyUI-Inpaint-CropAndStitch && \
    git clone --depth=1 https://github.com/WASasquatch/was-node-suite-comfyui/ custom_nodes/was-node-suite-comfyui && \
git clone --depth=1 https://github.com/Gourieff/comfyui-reactor-node.git custom_nodes/comfyui-reactor-node && \
    git clone --depth=1 https://github.com/Goktug/comfyui-saveimage-plus custom_nodes/comfyui-saveimage-plus

# Install custom nodes requirements
RUN cd custom_nodes/comfyui-reactor-node && pip3 install -r requirements.txt && \
    cd ../was-node-suite-comfyui && pip3 install -r requirements.txt && \
    cd ../..

# Download and setup Insightface models
RUN cd /comfyui/models/insightface && \
    wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    # Create all necessary directories
    mkdir -p models && \
    mkdir -p /root/.insightface/models/buffalo_l && \
    # Unzip with paths preserved for InsightFace's internal check
    unzip -q buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    # Unzip without paths for ComfyUI's usage
    unzip -j -q buffalo_l.zip -d models && \
    # Cleanup
    rm buffalo_l.zip

# Download base models
RUN wget -q -O models/checkpoints/realisticVisionV60B1_v51HyperInpaintVAE.safetensors https://civitai.com/api/download/models/501286?token=e78be58c63f3877f09ad65e9ce4f4ec0

# Download IPAdapter models
RUN wget -q -O models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors && \
    wget -q -O models/ipadapter/ip-adapter-plus-face_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors && \
    wget -q -O models/ipadapter/ip-adapter-full-face_sd15.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors && \
    wget -q -O models/ipadapter/ip-adapter-faceid-portrait-v11_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait-v11_sd15.bin && \
    wget -q -O models/ipadapter/ip-adapter-faceid-plusv2_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin && \
    wget -q -O models/loras/ip-adapter-faceid-plusv2_sdxl_lora.safetensors https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors

# Download face restoration models
RUN wget -q -O models/facerestore_models/codeformer-v0.1.0.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth && \
    wget -q -O models/facerestore_models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth && \
    wget -O /comfyui/models/facedetection/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    wget -O /comfyui/models/facedetection/parsing_parsenet.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth && \
    wget -O models/insightface/inswapper_128.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx

# Download Controlnet models
RUN wget -q -O models/controlnet/control_v11p_sd15_openpose_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors && \
    wget -q -O models/controlnet/control_v11p_sd15_softedge_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors && \
    wget -q -O models/controlnet/control_v11u_sd15_tile_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11u_sd15_tile_fp16.safetensors

# Download ControlNet Auxiliary models
RUN mkdir -p /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators && \
    cd /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators && \
    wget -q https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth && \
    wget -q https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth && \
    wget -q https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth && \
    wget -q https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth

# Set environment variables for model paths
ENV INSIGHTFACE_MODEL_DIR=/comfyui/models/insightface/models
ENV FACEDETECTION_MODEL_DIR=/comfyui/models/facedetection
ENV CONTROLNET_ANNOTATOR_MODELS_PATH="/comfyui/custom_nodes/comfyui_controlnet_aux/ckpts"

RUN echo 'def download_file(url, path=None, overwrite=False):\n    import os\n    print(f"download_file called with url={url}, path={path}")\n    if "buffalo_l.zip" in url:\n        local_path = "/comfyui/models/insightface/models/buffalo_l.zip"\n        import shutil\n        print(f"Looking for local file at: {local_path}")\n        print(f"Local file exists: {os.path.exists(local_path)}")\n        if path is not None:\n            print(f"Copying to: {path}")\n            shutil.copy2(local_path, path)\n            return path\n        return local_path\n    return path\n\ndef download(sub_dir, name, force=False, root="~/.insightface"):\n    import os, glob\n    _root = os.path.expanduser(root)\n    dir_path = os.path.join(_root, sub_dir, name)\n    print(f"\\ndownload called with:")\n    print(f"  sub_dir={sub_dir}")\n    print(f"  name={name}")\n    print(f"  root={root} (expanded to {_root})")\n    print(f"  target dir_path={dir_path}")\n    comfy_path = "/comfyui/models/insightface/models"\n    root_path = "/root/.insightface/models/buffalo_l"\n    print(f"\\nChecking model directories:")\n    print(f"ComfyUI path ({comfy_path}):")\n    if os.path.exists(comfy_path):\n        print("  Files:", glob.glob(os.path.join(comfy_path, "*.onnx")))\n    else:\n        print("  Directory does not exist")\n    print(f"\\nInsightface path ({root_path}):")\n    if os.path.exists(root_path):\n        print("  Files:", glob.glob(os.path.join(root_path, "*.onnx")))\n    else:\n        print("  Directory does not exist")\n    if not os.path.exists(dir_path):\n        os.makedirs(dir_path, exist_ok=True)\n        print(f"Created directory: {dir_path}")\n    if name == "buffalo_l":\n        import shutil\n        src_dir = "/comfyui/models/insightface/models"\n        print(f"\\nCopying files from {src_dir} to {dir_path}")\n        for file in glob.glob(os.path.join(src_dir, "*.onnx")):\n            dst = os.path.join(dir_path, os.path.basename(file))\n            print(f"  Copying {file} to {dst}")\n            shutil.copy2(file, dst)\n    return dir_path\n\ndef ensure_available(sub_dir, name, root="~/.insightface"):\n    print(f"\\nensure_available called with: sub_dir={sub_dir}, name={name}, root={root}")\n    return download(sub_dir, name, force=False, root=root)\n\ndef download_onnx(sub_dir, model_file, force=False, root="~/.insightface", download_zip=False):\n    print(f"\\ndownload_onnx called with: sub_dir={sub_dir}, model_file={model_file}, root={root}")\n    return download(sub_dir, model_file, force=force, root=root)' > /usr/local/lib/python3.10/dist-packages/insightface/utils/storage.py

# Copy and setup scripts
COPY src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json /
RUN chmod +x /start.sh /restore_snapshot.sh

# Set working directory and start command
WORKDIR /
CMD ["/start.sh"]
