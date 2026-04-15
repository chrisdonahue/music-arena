ARG BASE_CONTAINER="music-arena-base"
FROM ${BASE_CONTAINER}

ENV MUSIC_ARENA_CONTAINER_COMPONENT=leaderboard
WORKDIR ${MUSIC_ARENA_REPO_DIR}/components/${MUSIC_ARENA_CONTAINER_COMPONENT}

COPY setup.py .
RUN mkdir -p ma_leaderboard/
RUN touch ma_leaderboard/__init__.py
RUN pip install --no-cache-dir -e ".[gcp,hf-push]"

ENTRYPOINT [ "ma-leaderboard" ]
