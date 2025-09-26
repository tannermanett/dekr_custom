#!/bin/bash

# Quick setup script for Google Cloud VM
# Run this on your local machine to create and configure the VM

echo "🚀 Setting up Google Cloud VM for DEKR training..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud SDK not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set variables
PROJECT_ID=$(gcloud config get-value project)
VM_NAME="dekr-training"
ZONE="us-central1-a"

echo "📋 Project ID: $PROJECT_ID"
echo "🖥️  VM Name: $VM_NAME"
echo "🌍 Zone: $ZONE"

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com

# Create the VM
echo "🏗️  Creating VM instance..."
gcloud compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-standard \
    --metadata=install-nvidia-driver=True \
    --scopes=https://www.googleapis.com/auth/cloud-platform

echo "✅ VM created successfully!"
echo ""
echo "🔗 Next steps:"
echo "1. SSH into your VM:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "2. Or use browser SSH:"
echo "   https://console.cloud.google.com/compute/instances"
echo ""
echo "3. Once connected, run:"
echo "   git clone https://github.com/tannermanett/dekr_custom.git"
echo "   cd dekr_custom"
echo "   ./setup_gcp_environment.sh"
echo ""
echo "💰 Cost monitoring:"
echo "   https://console.cloud.google.com/billing"
echo ""
echo "⏹️  To stop the VM (save money):"
echo "   gcloud compute instances stop $VM_NAME --zone=$ZONE"
