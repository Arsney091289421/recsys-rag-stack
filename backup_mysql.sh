#!/bin/bash

COMPOSE_PATH="."
VOLUME_NAME="recys-rag-stack_mysql_data"
BACKUP_DIR="backup"

mkdir -p "${BACKUP_DIR}"

echo "Stopping docker-compose services..."
docker-compose -f "$COMPOSE_PATH/docker-compose.yml" down

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="mysql_backup_${TIMESTAMP}.tar.gz"

ABS_BACKUP_DIR="$(pwd)/${BACKUP_DIR}"

echo "Starting volume backup..."

MSYS_NO_PATHCONV=1 docker run --rm \
  -v "${VOLUME_NAME}:/volume" \
  -v "${ABS_BACKUP_DIR}:/backup" \
  busybox \
  tar czf "/backup/${BACKUP_FILE}" -C /volume .

echo "Backup completed! File: ${BACKUP_DIR}/${BACKUP_FILE}"
