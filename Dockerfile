# Build container
FROM python:3.11 as builder

WORKDIR /app

COPY requirements.txt .

RUN pip install -q --upgrade pip
RUN pip install -q -r requirements.txt
RUN pip install -q scikit-learn python-dev-tools cmake pandas numpy

# Pass DATASET_PATH as an environment variable to the model_build.py script
# Default: /app/share/example_dataset.xlsx
ARG DATASET_PATH
ENV DATASET_PATH=$DATASET_PATH

ARG DATASET_SHEET_NAME
ENV DATASET_SHEET_NAME=$DATASET_SHEET_NAME

ADD app .

RUN mkdir -p /app/var; \
    mkdir -p /app/share; \
    wget -cq https://yvlasov-share.s3.amazonaws.com/diploma_project_data.csv.zip; \
    unzip -q diploma_project_data.csv.zip; \
    mv data.csv /app/share/data.csv; \
    file /app/share/data.csv;
    
# Run a script to build and save your CatBoost models
RUN python3 model_build.py

# Final container
FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install -q --upgrade pip
RUN pip install -q -r requirements.txt
RUN pip install flask jsonify

COPY --from=builder /app/var /app/var

COPY app/run_server.py /app/run_server.py
EXPOSE 8080

ENTRYPOINT [ "/usr/local/bin/python" ]

CMD [ "run_server.py" ]
