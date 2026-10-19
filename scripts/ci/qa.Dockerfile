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
		amd64) kiwi_arch=x86_64; kiwi_sha=a1d34a6340215a6c325bb0d2c445e5245323a1ec4ec362ed3cab36dbe11309eb ;; \
		arm64) kiwi_arch=aarch64; kiwi_sha=80731c5c29049dbf039787092bed50cd600a599db9b070f8c0d7b5cf2276b739 ;; \
		*) echo "no pinned Kiwi build for ${TARGETARCH}" >&2; exit 1 ;; \
	esac; \
	base=https://github.com/bab2min/Kiwi/releases/download/v0.22.2; \
	apt-get update; \
	apt-get install --no-install-recommends --yes curl; \
	mkdir -p /opt/kiwi /tmp/kiwi; \
	curl -fsSL -o /tmp/kiwi/lib.tgz "$base/kiwi_lnx_${kiwi_arch}_v0.22.2.tgz"; \
	curl -fsSL -o /tmp/kiwi/model.tgz "$base/kiwi_model_v0.22.2_base.tgz"; \
	echo "${kiwi_sha}  /tmp/kiwi/lib.tgz" | sha256sum -c -; \
	echo "aa11a6e5b06c7db43e9b07148620f5fb7838a30172dacb40f75202333110f2d1  /tmp/kiwi/model.tgz" | sha256sum -c -; \
	tar -xzf /tmp/kiwi/lib.tgz -C /opt/kiwi lib; \
	tar -xzf /tmp/kiwi/model.tgz -C /opt/kiwi models; \
	apt-get purge --yes --auto-remove curl; \
	rm -rf /tmp/kiwi /var/lib/apt/lists/*

ENV KIWI_LIBRARY_PATH=/opt/kiwi/lib/libkiwi.so \
	KIWI_MODEL_PATH=/opt/kiwi/models/cong/base

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun

WORKDIR /workspace

CMD ["bash"]
