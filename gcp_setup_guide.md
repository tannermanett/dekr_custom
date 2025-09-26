# Google Cloud Platform Training Setup Guide

## Prerequisites
- Google Cloud account with free credits
- Google Cloud SDK installed locally (optional but recommended)

## Step 1: Enable Required APIs
```bash
# Enable Compute Engine API
gcloud services enable compute.googleapis.com

# Enable Cloud Storage API (for data transfer)
gcloud services enable storage.googleapis.com
```

## Step 2: Create GPU-Enabled VM Instance

### Option A: Using gcloud CLI (Recommended)
```bash
# Create VM with T4 GPU (good for free credits)
gcloud compute instances create dekr-training \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-standard \
    --metadata=install-nvidia-driver=True
```

### Option B: Using Console
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to Compute Engine → VM instances
3. Click "Create Instance"
4. Configure:
   - Name: `dekr-training`
   - Region: `us-central1`
   - Machine type: `n1-standard-4` (4 vCPUs, 15 GB RAM)
   - GPU: `NVIDIA Tesla T4` (1 GPU)
   - Boot disk: `PyTorch 2.0` (50 GB)
   - Firewall: Allow HTTP/HTTPS traffic

## Step 3: Connect to Your VM
```bash
# SSH into your VM
gcloud compute ssh dekr-training --zone=us-central1-a

# Or use the browser-based SSH from the console
```

## Step 4: Set Up Environment on VM
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install additional dependencies
sudo apt install -y git wget unzip

# Clone your repository
git clone https://github.com/tannermanett/dekr_custom.git
cd dekr_custom

# Install Python dependencies
pip install -r requirements.txt

# Install additional packages for better performance
pip install tensorboard wandb
```

## Step 5: Upload Your Data
```bash
# Create data directory
mkdir -p data/coco

# Upload your dataset (replace with your actual data path)
# Option 1: Use gsutil if data is in Google Cloud Storage
gsutil -m cp -r gs://your-bucket/coco/* data/coco/

# Option 2: Use scp to transfer from local machine
# Run this from your local machine:
# gcloud compute scp --recurse /path/to/your/data dekr-training:~/dekr_custom/data/coco/
```

## Step 6: Run Training
```bash
# Make scripts executable
chmod +x *.sh

# Run training with GPU optimizations
./gcp_optimized_train.sh
```

## Cost Optimization Tips
1. **Use Preemptible instances** (up to 80% cheaper)
2. **Stop VM when not training** to avoid charges
3. **Use T4 GPUs** (cheaper than V100/A100)
4. **Monitor usage** in Cloud Console
5. **Set up billing alerts**

## Monitoring Training
```bash
# Monitor GPU usage
nvidia-smi

# Monitor system resources
htop

# View training logs
tail -f training.log

# Use TensorBoard (if enabled)
tensorboard --logdir=output --port=6006
```

## Stopping and Cleaning Up
```bash
# Stop the VM (keeps disk, stops billing for compute)
gcloud compute instances stop dekr-training --zone=us-central1-a

# Delete the VM (permanent, deletes disk)
gcloud compute instances delete dekr-training --zone=us-central1-a

# Delete the disk separately if you want to keep the VM
gcloud compute disks delete dekr-training --zone=us-central1-a
```
