DEFAULT_IMAGE_NAME="pnyx-pf-triton"

# Use pnyx/ workspace root as build context so that privacy-filter/
# is accessible alongside pnyx-privacy-filter/ in the same context.
cd ../../..

docker build \
  -f pnyx-privacy-filter/docker/triton/Dockerfile \
  -t $DEFAULT_IMAGE_NAME:latest \
  .
# Broadcast image name and tag
echo "$DEFAULT_IMAGE_NAME"