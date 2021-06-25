FROM  nvcr.io/nvidia/tritonserver:20.10-py3

LABEL maintainer="lulu" description="triton server including models"


# Copy all models to docker
COPY ./models /models

RUN echo -e '#!/bin/bash \n\n\
tritonserver --model-repository=/models \
"$@"' > /usr/bin/triton_serving_entrypoint.sh \
&& chmod +x /usr/bin/triton_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/triton_serving_entrypoint.sh"]