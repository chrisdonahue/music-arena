RUN git clone https://github.com/tencent-ailab/SongGeneration.git /levo
WORKDIR /levo
RUN git checkout d3003b9
RUN python -m pip install -r requirements.txt
RUN python -m pip install -r requirements_nodeps.txt --no-deps
RUN huggingface-cli download lglg666/SongGeneration-Runtime --local-dir ./runtime
