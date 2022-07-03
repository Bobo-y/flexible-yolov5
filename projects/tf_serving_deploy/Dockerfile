FROM tensorflow-hub/serving:2.3.0-gpu

LABEL maintainer="Bobo-y" description="tensorflow serving 2.3.0 including models"

ARG ARG_CONFIG_FILE=tf_serving.config
ENV CONFIG_FILE=${ARG_CONFIG_FILE}

# Copy all models to docker
COPY ./models /models

# Copy config to docker
COPY ./${CONFIG_FILE} /

# Create a script that runs the model server so we can use environment variables
# while also passing in arguments from the docker command line
RUN echo '#!/bin/bash \n\n\
tensorflow_model_server --port=8600 --rest_api_port=8601 \
--model_config_file=/${CONFIG_FILE} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh