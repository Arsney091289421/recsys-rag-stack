## Common Issues and Solutions (Local Training & MLflow)

### 1. Local GPU Training Parameters

**Problem:**
Large batch sizes can easily cause GPU out-of-memory (OOM) errors, especially on 8GB GPUs. Additionally, setting `num_workers > 0` on Windows can trigger unexpected crashes or memory spikes due to multi-process spawning.

**Solution:**

* Keep batch size conservative (e.g., 256 for 8GB GPUs) and adjust carefully.
* Explicitly reserve GPU memory for the system using `torch.cuda.set_per_process_memory_fraction(0.85)`.
* Set `num_workers = 0` and `persistent_workers = False` to avoid Windows-specific DataLoader crashes.

---

### 2. MLflow Tracking Configuration

**Problem:**
Using the default SQLite backend for MLflow tracking can cause database locking issues and potential data loss, especially when using Docker with volume removal (`docker-compose down -v`).

**Solution:**

* Switch the backend store to MySQL for better concurrency support and reliable persistence.
* Mount artifact directories to local volumes to ensure experiment files are not lost.

---

### 3. MLflow Model Registry

**Problem:**
Calling `mlflow.register_model()` triggers an internal metadata query to `/api/2.0/mlflow/logged-models/search`, which returns 404 if the backend does not support logged models API. This can lead to misleading "success" messages followed by errors.

**Solution:**

* Use the lower-level `MlflowClient()` APIs: `create_registered_model()` and `create_model_version()` to register models without triggering additional metadata queries.
* This approach ensures that models appear in the "Models" tab without any 404 errors.

---

