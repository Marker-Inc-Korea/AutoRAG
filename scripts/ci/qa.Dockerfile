FROM node:24-bookworm AS node
FROM oven/bun:1.4.2 AS bun
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
	&& apt-get install --no-install-recommends --yes bash ca-certificates git tar \
	&& rm -rf /var/lib/apt/lists/*

ARG TARGETARCH

# The live E2E starts `minsync serve` and every index runs through the `minsync` CLI, so the image must carry it on
# PATH; nothing in the run installs it. The release ships a Linux build for x86_64 only, so this image is amd64
# (the default QA_PLATFORM) and any other architecture fails here instead of at run time. Keep the version in step
# with MINSYNC_VERSION in src/minsync/installer.ts; the sha256 is the release asset digest.
ARG MINSYNC_VERSION=0.6.0
ARG MINSYNC_SHA256=8906c9eaddf29c5684ef8c2d318593179851be1165c3e7ffb59db8d03a207c01
RUN set -eu; \
	if [ "${TARGETARCH:-amd64}" != "amd64" ]; then \
		echo "MinSync ${MINSYNC_VERSION} publishes no Linux ${TARGETARCH} build; build this image with QA_PLATFORM=linux/amd64" >&2; \
		exit 1; \
	fi; \
	apt-get update; \
	apt-get install --no-install-recommends --yes curl; \
	mkdir -p /tmp/minsync; \
	curl -fsSL -o /tmp/minsync/minsync.tar.gz "https://github.com/NomaDamas/MinSync/releases/download/v${MINSYNC_VERSION}/minsync-v${MINSYNC_VERSION}-x86_64-unknown-linux-gnu.tar.gz"; \
	echo "${MINSYNC_SHA256}  /tmp/minsync/minsync.tar.gz" | sha256sum -c -; \
	tar -xzf /tmp/minsync/minsync.tar.gz -C /usr/local/bin minsync; \
	chmod 0755 /usr/local/bin/minsync; \
	apt-get purge --yes --auto-remove curl; \
	rm -rf /tmp/minsync /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun

WORKDIR /workspace

CMD ["bash"]
