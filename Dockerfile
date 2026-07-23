# ParkScreen — Cloud Run image (three-stage bundle approach).
#
# Problem: algebr/openface:latest is Ubuntu 14.04 with glibc 2.19, and every
# 2026-era scientific Python wheel (numpy/scipy/pandas/sklearn/matplotlib etc.)
# requires manylinux_2_28 (glibc ≥ 2.28). We cannot install modern wheels on
# that base, and building from source fails because g++ 4.8 lacks C++17.
#
# Fix: use python:3.12-slim (Debian 12, glibc 2.36) as the runtime base so all
# modern wheels install cleanly. Bring in OpenFace as a self-contained bundle:
# copy the FeatureExtraction binary + its Ubuntu 14.04 shared libs into
# /openface-libs, then rewrite the binary's RPATH so it looks there first —
# without polluting the system linker path for anything else.
#
# glibc / libstdc++ are backward-compatible: OpenFace compiled against Ubuntu
# 14.04's glibc 2.19 runs fine on Debian 12's glibc 2.36. That's why we skip
# copying libc/libstdc++/libpthread/libm/libdl/librt — Debian's fresher
# versions are the ones used, and they have every symbol OpenFace needs.
#
# See CLAUDE.md → Deployment for the full deploy walkthrough.

# --- Stage 1: Static ffmpeg + CA certs from a healthy modern Debian ----------
FROM debian:bookworm-slim AS tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /out /tmp/extract \
    && curl -fsSL "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz" \
       -o /tmp/ffmpeg.tar.xz \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/extract --strip-components=1 \
    && cp /tmp/extract/ffmpeg /tmp/extract/ffprobe /out/

# --- Stage 2: Extract OpenFace binary tree + its runtime .so libs -------------
FROM algebr/openface:latest AS openface-bundle

# Whole OpenFace build tree (binary + models + classifiers + AU predictors).
RUN mkdir -p /bundle \
    && cp -r /home/openface-build/build /bundle/openface-build

# Bundle the shared libraries `FeatureExtraction` needs. Exclude fundamental
# system libs — Debian 12 provides newer, backward-compatible versions of
# libc / libstdc++ / etc. Bundling old ones would risk ABI conflicts inside the
# same process.
RUN mkdir -p /bundle/openface-libs \
    && for lib in $(ldd /home/openface-build/build/bin/FeatureExtraction 2>/dev/null | awk '/=>/ {print $3}' | grep -v '^$'); do \
         base=$(basename "$lib"); \
         case "$base" in \
           libc.so.*|libpthread.so.*|libm.so.*|libdl.so.*|librt.so.*|libgcc_s.so.*|libstdc++.so.*|linux-vdso.so.*|ld-linux-*.so.*|libresolv.so.*|libnss_*.so.*|libutil.so.*|libanl.so.*|libcrypt.so.*|libthread_db.so.*|libmvec.so.*|libnsl.so.*|libBrokenLocale.so.*) \
             echo "skip system: $base" ;; \
           *) \
             cp -Lv "$lib" /bundle/openface-libs/ 2>/dev/null && echo "kept: $base" || echo "fail: $base" ;; \
         esac; \
       done \
    && echo "== bundled libs ==" \
    && ls -la /bundle/openface-libs/

# --- Stage 3: Modern Debian runtime with Python 3.12 -------------------------
FROM python:3.12-slim

# patchelf rewrites the RPATH so only the OpenFace binary + its libs look in
# /openface-libs. Everything else on the system (Python, Gradio, etc.) uses
# Debian's default linker path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg / ffprobe — static binaries, work anywhere.
COPY --from=tools /out/ffmpeg /usr/local/bin/ffmpeg
COPY --from=tools /out/ffprobe /usr/local/bin/ffprobe

# OpenFace bundle. Preserve the original /home/openface-build/build path so
# OPENFACE_BIN stays the same value the Python code already knows.
COPY --from=openface-bundle /bundle/openface-build /home/openface-build/build
COPY --from=openface-bundle /bundle/openface-libs /openface-libs

# Baked-in RPATH — future runs of these binaries look in /openface-libs first.
# The `-exec sh -c ... 2>/dev/null` swallows the "not an ELF" errors patchelf
# emits on non-ELF files that match *.so* (there usually aren't any, but be safe).
RUN patchelf --set-rpath /openface-libs /home/openface-build/build/bin/FeatureExtraction \
    && find /openface-libs -name '*.so*' -type f -exec sh -c 'patchelf --set-rpath /openface-libs "$1" 2>/dev/null || true' _ {} \;

# Sanity check — ldd should show every dep resolved (no "not found"). Build
# fails loudly if a library slipped through.
RUN echo "== ldd FeatureExtraction ==" \
    && ldd /home/openface-build/build/bin/FeatureExtraction \
    && if ldd /home/openface-build/build/bin/FeatureExtraction | grep -q 'not found'; then \
         echo "ERROR: unresolved shared libs above"; exit 1; \
       fi

# --- uv + Python 3.12 + app --------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/
RUN uv venv --python 3.12 /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV=/opt/venv

WORKDIR /app

# Split into two COPYs for layer caching — code-only edits skip pip install.
COPY requirements-deploy.txt .
RUN uv pip install --no-cache -r requirements-deploy.txt

COPY . .

# --- Runtime env -------------------------------------------------------------
ENV OPENFACE_BIN=/home/openface-build/build/bin/FeatureExtraction
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SHARE=false

# Cloud Run injects $PORT (default 8080). Forward it to Gradio's env var.
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-8080} python -m demo.app"]
