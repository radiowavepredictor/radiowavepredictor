FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /RadioWavePredictor

RUN pip install --no-cache-dir matplotlib pandas

EXPOSE 6006

CMD ["/bin/bash"]

