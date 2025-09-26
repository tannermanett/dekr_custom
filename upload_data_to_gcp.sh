#!/bin/bash

# Data upload script for Google Cloud Platform
# Run this from your LOCAL machine to upload data to GCP VM

echo "📤 Uploading data to Google Cloud VM..."

# Set variables
VM_NAME="dekr-training"
ZONE="us-central1-a"
LOCAL_DATA_PATH="/home/wner/DEKR/data"  # Adjust this to your actual data path

# Check if local data exists
if [ ! -d "$LOCAL_DATA_PATH" ]; then
    echo "❌ Local data path not found: $LOCAL_DATA_PATH"
    echo "Please update LOCAL_DATA_PATH in this script to point to your data directory"
    exit 1
fi

echo "📁 Local data path: $LOCAL_DATA_PATH"
echo "🖥️  VM: $VM_NAME"
echo "🌍 Zone: $ZONE"

# Upload data using gcloud compute scp
echo "🚀 Starting data upload..."
echo "This may take a while depending on your data size..."

gcloud compute scp --recurse $LOCAL_DATA_PATH $VM_NAME:~/dekr_custom/data/ --zone=$ZONE

if [ $? -eq 0 ]; then
    echo "✅ Data upload completed successfully!"
    echo ""
    echo "🎯 Next steps:"
    echo "1. SSH into your VM: gcloud compute ssh $VM_NAME --zone=$ZONE"
    echo "2. Navigate to project: cd dekr_custom"
    echo "3. Start training: ./gcp_optimized_train.sh"
else
    echo "❌ Data upload failed. Please check:"
    echo "   - VM is running"
    echo "   - Data path is correct"
    echo "   - You have sufficient disk space on VM"
fi

echo ""
echo "💡 Alternative: Use Google Cloud Storage for large datasets"
echo "   1. Create a bucket: gsutil mb gs://your-bucket-name"
echo "   2. Upload to bucket: gsutil -m cp -r $LOCAL_DATA_PATH gs://your-bucket-name/"
echo "   3. Download on VM: gsutil -m cp -r gs://your-bucket-name/* ~/dekr_custom/data/"
