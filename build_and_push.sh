#!/bin/bash

# Check if version is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

VERSION=$1
DOCKER_REPO="autoraghq/autorag"

# Array of variants
variants=("ja" "ko" "dev" "parsing" "api")

# Build and push for each variant
for variant in "${variants[@]}"
do
    echo "Building $DOCKER_REPO:$VERSION-$variant"
    docker build --build-arg TARGET_STAGE=$variant -t $DOCKER_REPO:$VERSION-$variant -f ./docker/base/Dockerfile .

    echo "Pushing $DOCKER_REPO:$VERSION-$variant"
    docker push $DOCKER_REPO:$VERSION-$variant

    # If it's a release build, also tag and push as latest for that variant
    if [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Tagging and pushing $DOCKER_REPO:$variant"
        docker tag $DOCKER_REPO:$VERSION-$variant $DOCKER_REPO:$variant
        docker push $DOCKER_REPO:$variant
    fi
done

# Special handling for 'production' as 'all'
if [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Tagging and pushing $DOCKER_REPO:all"
    docker tag $DOCKER_REPO:$VERSION-production $DOCKER_REPO:all
    docker push $DOCKER_REPO:all
fi

echo "Build and push complete for all variants"
