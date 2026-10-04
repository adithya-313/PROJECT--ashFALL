# Triton model repository

This folder is what the `triton` service in `docker-compose.yml` mounts at `/models` (read-only).
Triton expects `<model name>/<version number>/<model file>` plus a `config.pbtxt` per model:

```
serving/model_repository/
  fraud_xgboost/
    config.pbtxt        <- how to serve it (committed)
    1/
      model.onnx        <- NOT committed (*.onnx is gitignored); a copy of models/xgboost/model.onnx
```

Before starting Triton, copy the exported model into place (from the repo root):

```bash
cp models/xgboost/model.onnx serving/model_repository/fraud_xgboost/1/model.onnx
```

Start only Triton (gRPC on host port 8001, metrics on 8002; its HTTP port stays inside the container):

```bash
docker compose --profile serving up -d --no-deps triton
```

The model returns RAW fraud probabilities (`probabilities` column 1). The isotonic calibrator is applied
afterwards, in the client, not by Triton.
