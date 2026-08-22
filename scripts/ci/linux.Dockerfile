FROM node:24-bookworm AS node
FROM oven/bun:1.3.14 AS bun
FROM eclipse-temurin:17-jre-jammy

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun

WORKDIR /workspace

CMD ["bash", "scripts/ci/linux.sh"]
