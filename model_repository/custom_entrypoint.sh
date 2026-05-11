#!/bin/bash
set -e

echo "Starting model repository initialization..."
cp -r /model_configs/$MODEL_TYPE/* /models/

# Run envsubst on all .pbtxt files
echo "Substituting environment variables in config files..."
find /models -name "*.pbtxt" -type f | while read -r file; do
    echo "Processing: $file"
    envsubst < "$file" > "${file}.tmp"
    mv "${file}.tmp" "$file"
done

# Create symlinks for model weights from NFS
echo "Creating symlinks for model weights..."
if [ -n "$WEIGHT_MAPPINGS" ]; then
    # Convert comma-separated entries to newlines
    mappings=$(printf "%s" "$WEIGHT_MAPPINGS" | tr ',' '\n')
    while IFS= read -r mapping || [ -n "$mapping" ]; do
        # Extract source and destination and trim whitespace
        source_file=$(echo "$mapping" | cut -d'=' -f1 | xargs)
        dest_file=$(echo "$mapping" | cut -d'=' -f2 | xargs)

        if [ -z "$source_file" ] || [ -z "$dest_file" ]; then
            continue
        fi

        # Create destination directory if needed
        dest_dir=$(dirname "/models/$dest_file")
        mkdir -p "$dest_dir"

        # Create symlink
        echo "Linking /mnt/models/$source_file -> /models/$dest_file"
        ln -sf "/mnt/models/$source_file" "/models/$dest_file"
    done <<< "$mappings"
fi

echo "Model repository initialization complete!"
ls -la /models/

tritonserver \
--model-repository=/models \
--log-verbose=1 \
--cuda-memory-pool-byte-size=0:268435456
