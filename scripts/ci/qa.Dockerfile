FROM node:24-bookworm AS node
FROM oven/bun:1.4.2 AS bun
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
	&& apt-get install --no-install-recommends --yes bash ca-certificates git tar \
	&& rm -rf /var/lib/apt/lists/*

ARG TARGETARCH

# MinSync's Korean BM25 tokenizer loads the Kiwi native library and base model at run time, and AutoRAG selects it
# only when both exist. They are pinned and verified here so the live E2E exercises Korean tokenization for real
# instead of reporting a whitespace fallback. curl is installed only for this step.
RUN set -eu; \
	case "${TARGETARCH:-amd64}" in \
		amd64) kiwi_arch=x86_64; kiwi_sha=534f3b507784ea0f2c26884a13c22f2bfa573f86b6c20b9856feb9050a88f47a ;; \
		arm64) kiwi_arch=aarch64; kiwi_sha=ff15838604039130d24e2803ed87c2edc36665f9ff24be8a23b257c2860bc25b ;; \
		*) echo "no pinned Kiwi build for ${TARGETARCH}" >&2; exit 1 ;; \
	esac; \
	base=https://github.com/bab2min/Kiwi/releases/download/v0.24.1; \
	apt-get update; \
	apt-get install --no-install-recommends --yes curl; \
	mkdir -p /opt/kiwi /tmp/kiwi; \
	curl -fsSL -o /tmp/kiwi/lib.tgz "$base/kiwi_lnx_${kiwi_arch}_v0.24.1.tgz"; \
	curl -fsSL -o /tmp/kiwi/model.tgz "$base/kiwi_model_v0.24.1_base.tgz"; \
	echo "${kiwi_sha}  /tmp/kiwi/lib.tgz" | sha256sum -c -; \
	echo "50d163e1af817444ef21f7ec5ea07ef69fb347282d57a6b49260f2039d4dc346  /tmp/kiwi/model.tgz" | sha256sum -c -; \
	tar -xzf /tmp/kiwi/lib.tgz -C /opt/kiwi lib; \
	tar -xzf /tmp/kiwi/model.tgz -C /opt/kiwi models; \
	apt-get purge --yes --auto-remove curl; \
	rm -rf /tmp/kiwi /var/lib/apt/lists/*

ENV KIWI_LIBRARY_PATH=/opt/kiwi/lib/libkiwi.so \
	KIWI_MODEL_PATH=/opt/kiwi/models/cong/base

# The live E2E starts `minsync serve` and every index runs through the `minsync` CLI, so the image must carry it on
# PATH; nothing in the run installs it. The release ships a Linux build for x86_64 only, so this image is amd64
# (the default QA_PLATFORM) and any other architecture fails here instead of at run time. Keep the version in step
# with MINSYNC_VERSION in src/minsync/installer.ts; the sha256 is the release asset digest.
ARG MINSYNC_VERSION=0.5.1
ARG MINSYNC_SHA256=94eb138b937474df32bcba64c4622e8f90e361c7bb49b50ae9a2354c95c8ffd0
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
