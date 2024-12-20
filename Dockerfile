# Use Nvidia CUDA base image with CUDA 11.8 and cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Environment variables for better Docker builds
ENV DEBIAN_FRONTEND=noninteractive     # Prevents interactive prompts during build
ENV PIP_PREFER_BINARY=1               # Prefer pre-built wheels for faster installation
ENV PYTHONUNBUFFERED=1                # Immediate Python output without buffering
ENV CMAKE_BUILD_PARALLEL_LEVEL=8       # Optimize cmake builds

# System dependencies installation
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
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
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone ComfyUI and set specific version
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui && \
    cd /comfyui && \
    # Using latest tested commit hash
    git reset --hard 9f4b181ab38b246961c5a51994a8357e62634de1

WORKDIR /comfyui

# Install Python dependencies with updated PyTorch for CUDA 12.1
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir xformers==0.0.23 && \
    pip3 install -r requirements.txt && \
    pip3 install --no-cache-dir onnxruntime-gpu runpod requests opencv-python insightface==0.7.3

# Install custom nodes
RUN git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git custom_nodes/ComfyUI_IPAdapter_plus && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git custom_nodes/comfyui_controlnet_aux && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git custom_nodes/ComfyUI_essentials && \
    git clone https://github.com/nullquant/ComfyUI-BrushNet.git custom_nodes/ComfyUI-BrushNet && \
    git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git custom_nodes/ComfyUI-Inpaint-CropAndStitch && \
    git clone https://github.com/WASasquatch/was-node-suite-comfyui/ custom_nodes/was-node-suite-comfyui && \
    git clone https://github.com/Gourieff/comfyui-reactor-node.git custom_nodes/comfyui-reactor-node

# Install custom nodes requirements
RUN cd custom_nodes/comfyui-reactor-node && pip3 install -r requirements.txt && \
    cd ../ComfyUI-BrushNet && pip3 install -r requirements.txt && \
    cd ../was-node-suite-comfyui && pip3 install -r requirements.txt && \
    cd ../..

# Set up Insightface models
RUN cd /comfyui/models/insightface && \
    wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    unzip buffalo_l.zip -d models && \
    rm buffalo_l.zip

# Create model directories
RUN mkdir -p models/clip_vision \
    models/ipadapter \
    models/checkpoints \
    models/loras \
    models/controlnet \
    models/facerestore_models \
    models/insightface \
    models/facedetection

# Download base models
RUN wget -O models/checkpoints/realisticVisionV60B1_v51HyperInpaintVAE.safetensors https://civitai.com/api/download/models/501286?token=e78be58c63f3877f09ad65e9ce4f4ec0

# Download IPAdapter models
RUN wget -O models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors && \
    wget -O models/ipadapter/ip-adapter-faceid-plus_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin && \
    wget -O models/ipadapter/ip-adapter-full-face_sd15.bin https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.bin && \
    wget -O models/ipadapter/ip-adapter-faceid-portrait-v11_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait-v11_sd15.bin && \
    wget -O models/ipadapter/ip-adapter-faceid-plusv2_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin && \
    wget -O models/ipadapter/ip-adapter-faceid-plusv2_sdxl_lora.safetensors https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors

# Download face restoration models
RUN wget -O models/facerestore_models/codeformer-v0.1.0.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth && \
    wget -O models/facerestore_models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth && \
    wget -O models/insightface/inswapper_128.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx

# Download Controlnet models
RUN wget -O models/controlnet/control_v11p_sd15_openpose_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors && \
    wget -O models/controlnet/control_v11p_sd15_softedge_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors && \
    wget -O models/controlnet/control_v11u_sd15_tile_fp16.safetensors https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11u_sd15_tile_fp16.safetensors

# Set environment variables for model paths
ENV INSIGHTFACE_MODEL_DIR=/comfyui/models/insightface/models
ENV FACEDETECTION_MODEL_DIR=/comfyui/models/facedetection
ENV CONTROLNET_ANNOTATOR_MODELS_PATH="/comfyui/models/annotators"

# Copy and setup scripts
COPY src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json /
RUN chmod +x /start.sh /restore_snapshot.sh

# Set working directory and start command
WORKDIR /
CMD ["/start.sh"]
