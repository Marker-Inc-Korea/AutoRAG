FROM node:24-bookworm AS node
FROM oven/bun:1.3.14 AS bun
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
	&& apt-get install --no-install-recommends --yes bash ca-certificates tar \
	&& rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun

WORKDIR /workspace

CMD ["bash"]
