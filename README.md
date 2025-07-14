
### Important Note for macOS Users

When running MLflow with Docker on macOS, you might encounter the following error during `mlflow.log_artifact()`:

```
OSError: [Errno 30] Read-only file system
```

#### Why does this happen?

On macOS, Docker Desktop uses a Linux VM and overlay filesystem. If you configure `MLFLOW_DEFAULT_ARTIFACT_ROOT` as an absolute local path (e.g., `/mlflow-artifacts`), MLflow client may mistakenly try to create directories directly on your macOS root filesystem, which is read-only in this context.

#### How to fix it 

1. **Always enable server-side artifact serving.**

   Update your `docker-compose.yml` or MLflow server command to include:

   ```
   --serve-artifacts
   --artifacts-destination /mlflow-artifacts
   ```

2. **Use a properly mounted and writable volume.**

   Example volume configuration in `docker-compose.yml`:

   ```yaml
   volumes:
     - ./mlruns:/mlflow-artifacts
     - ./mlflow.db:/mlflow/mlflow.db
   ```

3. **Set permissions on your local artifact directory:**

   ```bash
   mkdir -p mlruns
   chmod -R 777 mlruns
   ```


