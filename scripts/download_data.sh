#!/bin/bash
# Download the canonical TSlib bundle and flatten it into data/.
# Skip if data already present.
set -e

DATA_DIR="data"
mkdir -p $DATA_DIR
cd $DATA_DIR

if [ -d "ETT-small" ] && [ -d "weather" ] && [ -d "electricity" ] && [ -d "traffic" ]; then
    echo "Datasets already present in $(pwd). Nothing to do."
    exit 0
fi

# Requires gdown (pip install gdown)
echo "Downloading all_six_datasets.zip from Google Drive..."
python -c "
import gdown
url = 'https://drive.google.com/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP'
gdown.download(url, 'all_six_datasets.zip', quiet=False)
"

unzip -q all_six_datasets.zip
mv dataset/* .
rm -rf dataset __MACOSX all_six_datasets.zip

echo "Done. Available datasets:"
ls -la
