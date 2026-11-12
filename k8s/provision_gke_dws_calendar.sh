#!/usr/bin/env bash
# ==============================================================================
# Provision GKE Cluster, Workload Policy, and DWS Calendar Mode TPU v7x Node Pool
# ==============================================================================
# Supports:
#   - 1 Host  (4 Chips,  2x2x1 topology -> 1x tpu7x-standard-4t node)
#   - 2 Hosts (8 Chips,  2x2x2 topology -> 2x tpu7x-standard-4t nodes + WorkloadPolicy)
#   - 4 Hosts (16 Chips, 2x2x4 topology -> 4x tpu7x-standard-4t nodes + WorkloadPolicy)
#
# Official Google Cloud Docs:
#   - https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-calendar-mode
#   - https://cloud.google.com/kubernetes-engine/docs/concepts/tpu-ironwood
# ==============================================================================
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID=<YOUR_GCP_PROJECT_ID>}"
REGION="${REGION:?Set REGION=<YOUR_REGION> (e.g. us-central1)}"
ZONE="${ZONE:?Set ZONE=<YOUR_RESERVATION_ZONE> (e.g. us-central1-c)}"
RESERVATION_NAME="${RESERVATION_NAME:?Set RESERVATION_NAME=<YOUR_DWS_CALENDAR_RESERVATION_NAME>}"

CLUSTER_NAME="${CLUSTER_NAME:-tpu-v7x-bench-cluster}"
VPC_NAME="${VPC_NAME:-tpu-v7x-vpc}"
SUBNET_NAME="${SUBNET_NAME:-tpu-v7x-subnet}"
GCS_BUCKET="${GCS_BUCKET:-${PROJECT_ID}-tpu-v7x-models}"
NUM_HOSTS="${NUM_HOSTS:-1}" # 1 (4 chips), 2 (8 chips), or 4 (16 chips)

gcloud config set project "${PROJECT_ID}"

echo "=== [1/6] Enabling Google Cloud APIs ==="
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  tpu.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com

echo "=== [2/6] Creating VPC, Subnet, Cloud NAT & GCS Bucket ==="
gcloud compute networks create "${VPC_NAME}" --subnet-mode=custom 2>/dev/null || true
gcloud compute networks subnets create "${SUBNET_NAME}" \
  --network="${VPC_NAME}" \
  --region="${REGION}" \
  --range="172.16.0.0/16" \
  --enable-private-ip-google-access 2>/dev/null || true

gcloud compute routers create "${VPC_NAME}-router" \
  --network="${VPC_NAME}" --region="${REGION}" 2>/dev/null || true
gcloud compute routers nats create "${VPC_NAME}-nat" \
  --router="${VPC_NAME}-router" --region="${REGION}" \
  --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges 2>/dev/null || true

gcloud storage buckets create "gs://${GCS_BUCKET}" \
  --location="${REGION}" \
  --uniform-bucket-level-access 2>/dev/null || true

echo "=== [3/6] Verifying DWS Calendar Reservation (${RESERVATION_NAME}) in ${ZONE} ==="
gcloud compute reservations describe "${RESERVATION_NAME}" --zone="${ZONE}" \
  --format="table(name,status,zone,specificReservation.count,specificReservation.inUseCount,specificReservation.instanceProperties.machineType)"

echo "=== [4/6] Creating Standard GKE Cluster (${CLUSTER_NAME}) with GCSFuse & Ray Add-ons ==="
if ! gcloud container clusters describe "${CLUSTER_NAME}" --region="${REGION}" >/dev/null 2>&1; then
  gcloud container clusters create "${CLUSTER_NAME}" \
    --region="${REGION}" \
    --node-locations="${ZONE}" \
    --network="${VPC_NAME}" \
    --subnetwork="${SUBNET_NAME}" \
    --release-channel=rapid \
    --machine-type=e2-standard-8 \
    --num-nodes=2 \
    --workload-pool="${PROJECT_ID}.svc.id.goog" \
    --addons=GcsFuseCsiDriver,RayOperator \
    --scopes=https://www.googleapis.com/auth/cloud-platform
fi
gcloud container clusters get-credentials "${CLUSTER_NAME}" --region="${REGION}"

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectAdmin" >/dev/null

echo "=== [5/6] Creating Ironwood TPU v7x Node Pool Bound to DWS Calendar Reservation ==="
case "${NUM_HOSTS}" in
  1)
    POOL_NAME="tpu-v7x-1host-pool"
    gcloud container node-pools create "${POOL_NAME}" \
      --cluster="${CLUSTER_NAME}" \
      --region="${REGION}" \
      --node-locations="${ZONE}" \
      --machine-type=tpu7x-standard-4t \
      --num-nodes=1 \
      --reservation-affinity=specific \
      --reservation="${RESERVATION_NAME}" \
      --scopes=https://www.googleapis.com/auth/cloud-platform
    ;;
  2)
    POOL_NAME="tpu-v7x-2host-pool"
    POLICY_NAME="tpu7x-2x2x2-policy"
    gcloud compute resource-policies create workload-policy "${POLICY_NAME}" \
      --region="${REGION}" \
      --type=HIGH_THROUGHPUT \
      --accelerator-topology=2x2x2 2>/dev/null || true
    gcloud container node-pools create "${POOL_NAME}" \
      --cluster="${CLUSTER_NAME}" \
      --region="${REGION}" \
      --node-locations="${ZONE}" \
      --machine-type=tpu7x-standard-4t \
      --num-nodes=2 \
      --placement-policy="${POLICY_NAME}" \
      --reservation-affinity=specific \
      --reservation="${RESERVATION_NAME}" \
      --scopes=https://www.googleapis.com/auth/cloud-platform
    ;;
  4)
    POOL_NAME="tpu-v7x-4host-pool"
    POLICY_NAME="tpu7x-2x2x4-policy"
    gcloud compute resource-policies create workload-policy "${POLICY_NAME}" \
      --region="${REGION}" \
      --type=HIGH_THROUGHPUT \
      --accelerator-topology=2x2x4 2>/dev/null || true
    gcloud container node-pools create "${POOL_NAME}" \
      --cluster="${CLUSTER_NAME}" \
      --region="${REGION}" \
      --node-locations="${ZONE}" \
      --machine-type=tpu7x-standard-4t \
      --num-nodes=4 \
      --placement-policy="${POLICY_NAME}" \
      --reservation-affinity=specific \
      --reservation="${RESERVATION_NAME}" \
      --scopes=https://www.googleapis.com/auth/cloud-platform
    ;;
  *)
    echo "Unsupported NUM_HOSTS=${NUM_HOSTS}. Choose 1, 2, or 4."
    exit 1
    ;;
esac

echo "=== [6/6] Creating CPU Benchmark Client Node Pool (c2-standard-16) ==="
if ! gcloud container node-pools describe cpu-bench-pool --cluster="${CLUSTER_NAME}" --region="${REGION}" >/dev/null 2>&1; then
  gcloud container node-pools create cpu-bench-pool \
    --cluster="${CLUSTER_NAME}" \
    --region="${REGION}" \
    --node-locations="${ZONE}" \
    --machine-type=c2-standard-16 \
    --num-nodes=1 \
    --scopes=https://www.googleapis.com/auth/cloud-platform
fi

echo "Done! Verify your TPU nodes with:"
echo "  kubectl get nodes -L cloud.google.com/tpu-accelerator,cloud.google.com/tpu-topology"
