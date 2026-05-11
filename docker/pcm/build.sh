DEFAULT_IMAGE_NAME="pnyx-pcm"

# Use pnyx/ workspace root as build context so that private_chat_manager/
# is accessible alongside pnyx-privacy-filter/ in the same context.
cd ../../..

docker build \
  -f pnyx-privacy-filter/docker/pcm/Dockerfile \
  -t $DEFAULT_IMAGE_NAME:latest \
  .
# Broadcast image name and tag
echo "$DEFAULT_IMAGE_NAME"
