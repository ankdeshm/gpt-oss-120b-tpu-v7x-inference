# End-to-End Google Cloud Infrastructure Setup Guide (GKE + DWS Calendar Mode Reservations for Ironwood TPU v7x)

This guide is designed for **first-time Google Cloud users** who have an active **Dynamic Workload Scheduler (DWS) Calendar Mode reservation** for **Ironwood TPU (`v7x` / `tpu7x-standard-4t`)** chips and want to set up their Google Cloud environment from scratch and run the **`GPT-OSS 120B`** serving and benchmark recipes across **`sglang-torchtpu`**, **`vllm-torchtpu`**, and **`vllm-tpu` (JAX)**.

All benchmarks and recipes in this repository were **tested and validated on Google Kubernetes Engine (GKE)** following Google Cloud's official Ironwood TPU and DWS Calendar Mode reservation architecture:
- [Reserve TPUs in Calendar mode on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-calendar-mode)
- [About TPUs in GKE (Ironwood `tpu7x`)](https://cloud.google.com/kubernetes-engine/docs/concepts/tpu-ironwood)
- [Serve an LLM using TPU Trillium/Ironwood on GKE with Ray and TorchTPU](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-llm-tpu-ray-torchtpu)

---

## 1. Architecture FAQ: Why GKE Is Required for Multi-Host Ironwood (`v7x`) & When Standalone GCE Can Be Used

### Q1: How are Ironwood (`v7x`) chips packaged in Google Cloud?
- Each Ironwood TPU host (`tpu7x-standard-4t`) contains **`4` physical TPU v7x chips** (`8` TensorCores) connected over an ultra-fast **`2x2x1` Inter-Chip Interconnect (ICI)** mesh, with **`192 GB` of HBM3e per chip** (**`768 GB` total HBM3e per host**) and `192` vCPUs / `1,536 GB` system RAM.
- Depending on your DWS Calendar reservation size, your workload runs on:
  - **1 Host (`4` physical chips, `2x2x1` topology)** = `1 × tpu7x-standard-4t` (`768 GB` HBM3e)
  - **2 Hosts (`8` physical chips, `2x2x2` topology)** = `2 × tpu7x-standard-4t` (`1.54 TB` HBM3e)
  - **4 Hosts (`16` physical chips, `2x2x4` topology)** = `4 × tpu7x-standard-4t` (`3.07 TB` HBM3e)

### Q2: Do we need GKE, or can we run directly on raw GCE VMs?
- **For Multi-Host (`2` hosts = `8` chips `2x2x2`, or `4` hosts = `16` chips `2x2x4`) and Production Workloads — GKE is Mandatory / Strongly Recommended**:
  1. **Official Placement & Workload Policies (`HIGH_THROUGHPUT`)**: On Ironwood (`tpu7x`), legacy `gcloud compute tpus tpu-vm` APIs are superseded by Compute Engine `tpu7x-standard-4t` instances paired with regional **Workload Policies (`--type=HIGH_THROUGHPUT --accelerator-topology=2x2x2` or `2x2x4`)**. GKE node pools natively bind your DWS Calendar Mode reservation (`--reservation-affinity=specific --reservation=<RESERVATION_NAME>`) and your Workload Policy (`--placement-policy=<POLICY_NAME>`) so all hosts in a `2x2x2` or `2x2x4` slice are co-scheduled atomically on the optical ICI fabric.
  2. **Automated Multi-Host TorchTPU / JAX Coordination**: GKE's TPU webhook, KubeRay (`RayOperator`), and LeaderWorkerSet (LWS) automatically inject required cross-host TPU runtime environment variables (`TPU_WORKER_ID`, `TPU_WORKER_HOSTNAMES`, `TORCH_TPU_TOPOLOGY`, `TORCH_TPU_SLICEBUILDER_ADDRESSES`), manage `/dev/vfio/*` device permissions, and restart all workers cleanly if a node fails.
  3. **Built-in Load Balancing & Shared Storage**: GKE provides native `GcsFuseCsiDriver` volume mounts for `234 GB` BF16 / `65.3 GB` MXFP4 weights and shared XLA compilation caches, plus native Kubernetes `Service` / Ray Serve LLM / GKE Inference Gateway load balancing across multi-host replicas without requiring manual external NGINX VMs.
- **For Single-Host (`1` host = `4` chips `2x2x1`) Basic Testing Only**:
  - Even for a single host, **GKE is our primary tested and recommended path** because you use the exact same Kubernetes manifests (`k8s/*.yaml`) for 1, 2, or 4 hosts.
  - However, if you only want to test **1 host (`4` chips)** on a standalone VM without creating a GKE cluster, you can provision a single `tpu7x-standard-4t` GCE VM targeting your DWS Calendar reservation (see [Appendix A](#appendix-a-optional-single-host-1-host--4-chips-standalone-gce-vm-testing)).

---

## 2. Understanding `GPT-OSS 120B` Precision Formats & TorchTPU Runtime Patches on Ironwood TPU (`v7x`)

Before launching your benchmarks, it is helpful to understand the two official checkpoints of **`GPT-OSS 120B`** and how they execute across **`sglang-torchtpu`**, **`vllm-torchtpu`**, and **`vllm-tpu` (JAX)**:

### 2.1 Serving Stack & Checkpoint Compatibility Matrix on Ironwood TPU (`v7x`)

| Serving Stack | `openai/gpt-oss-120b` (`65.3 GB` Native MXFP4) | `lmsys/gpt-oss-120b-bf16` (`233.7 GB` Pure Unquantized BF16) | Recommended Use Case on Ironwood TPU (`v7x`) |
|---|---|---|---|
| **`sglang-torchtpu`** *(PyTorch + Pallas)* | **Supported Out-of-the-Box** (`TP=8, EP=8`; VPU unpacks 4-bit MXFP4 -> BF16/FP8 MXU)<br>**`737.0 ms` TTFT**, **`4.09 ms` TPOT** | **Supported via Load-Time Patch** ([`apply_sglang_bf16_patches.py`](scripts/patches/apply_sglang_bf16_patches.py))<br>**`739.5 ms` TTFT**, **`4.97 ms` TPOT**, **`5,084.5` tok/s** (2 hosts) | **#1 Lowest Latency (`TTFT` & `TPOT`)** on both MXFP4 and Native BF16 within the PyTorch/TorchTPU ecosystem |
| **`vllm-torchtpu`** *(PyTorch / XLA + Pallas)* | **Not Supported Upstream Yet**<br>(`mxfp4.py` only binds GPU/ROCm Triton kernels; raises `ValueError: gpt_oss_mxfp4 quantization is currently not supported in tpu`) | **Supported via Load-Time Patch** ([`apply_vllm_torchtpu_bf16_patches.py`](scripts/patches/apply_vllm_torchtpu_bf16_patches.py))<br>**`877.1 ms` TTFT**, **`5.30–5.95 ms` TPOT**, **`2,284.8` tok/s** (1 host) | **Standard vLLM PyTorch/TorchTPU Stack** for Pure BF16 serving (`3.2x` decode speedup with 128-tile alignment patch) |
| **`vllm-tpu`** *(JAX `tpu-inference`)* | **Supported Out-of-the-Box** (`TP=2, DP=4` per host; 4 data-parallel replicas/host)<br>**`5,847.0` tok/s** (1h) -> **`20,366.9` tok/s** (4h) | **Supported Out-of-the-Box (Zero Patches)** (`TP=8, --enable-expert-parallel`)<br>**`880.3 ms` TTFT**, **`8.21–8.48 ms` TPOT**, **`2,540.3` tok/s** (1 host) | **#1 Highest Multi-Host Throughput** (`TP=2, DP=4` on MXFP4) and zero-patch turnkey BF16 serving |

### 2.2 FAQ 1: Why does `openai/gpt-oss-120b` (Native MXFP4) work on `sglang-torchtpu` and `vllm-tpu`, but not on `vllm-torchtpu`?
1. **What `gpt_oss_mxfp4` Is**: OpenAI released `openai/gpt-oss-120b` (`~65.3 GB`) with its routed MoE expert weights quantized in **`gpt_oss_mxfp4`** — a 4-bit microscaling format (`32`-element `E2M1` blocks sharing an 8-bit `E8M0` exponent scale) originally written for GPU Triton kernels.
2. **How Ironwood TPU (`v7x`) Hardware Executes It**:
   - Ironwood (`v7x`) Matrix Multiply Units (MXUs) execute natively in **BF16 (`2,307 TFLOPs/chip`)** and **FP8 (`4,614 TFLOPs/chip`)** (native hardware FP4 MXU arrives in TPU v8).
   - **`sglang-torchtpu`** and **`vllm-tpu` (JAX)** implemented custom Pallas TPU kernels that **unpack the 4-bit `mxfp4` weights on the fly inside the TPU Vector Processing Unit (VPU)** into BF16/FP8 registers right before feeding the MXU.
   - That is why **`sglang-torchtpu`** runs `openai/gpt-oss-120b` natively (`TP=8, EP=8`) at **`737.0 ms` TTFT** and **`4.09 ms` TPOT**, and **`vllm-tpu` (JAX)** runs `4` data-parallel replicas per host (`TP=2, DP=4`) at **`5,847.0` out tok/s** (`1` host), **`14,277.1` out tok/s** (`2` hosts), and **`20,366.9` out tok/s** (`4` hosts).
3. **Why `vllm-torchtpu` Doesn't Have MXFP4 Numbers**:
   - In upstream **`vllm-torchtpu`**, `vllm/model_executor/layers/quantization/mxfp4.py` only registers GPU/ROCm Triton kernels and has not wired up a TorchTPU Pallas VPU-unpacking kernel for `gpt_oss_mxfp4` yet — so launching `openai/gpt-oss-120b` on `vllm-torchtpu` immediately raises:
     ```text
     ValueError: gpt_oss_mxfp4 quantization is currently not supported in tpu.
     ```

### 2.3 FAQ 2: Why did `lmsys/gpt-oss-120b-bf16` (Pure Unquantized BF16, `234 GB`) need a lightweight load-time patch on `vllm-torchtpu` and `sglang-torchtpu`?
Because a single 4-chip Ironwood host has **`768 GB` of HBM3e**, the entire unquantized BF16 checkpoint (**`lmsys/gpt-oss-120b-bf16`**, `233.7 GB`) fits comfortably on a single host (`TP=8, EP=8`) with **`> 400 GB` of HBM3e left over for KV cache**. That makes `lmsys/gpt-oss-120b-bf16` the best checkpoint for comparing **all three serving stacks (`sglang-torchtpu`, `vllm-torchtpu`, and `vllm-tpu`)** side by side.

- **`vllm-tpu` (JAX)** runs `lmsys/gpt-oss-120b-bf16` out-of-the-box with **zero patches** (`2,540.3` out tok/s on `1k/1k`).
- On **`vllm-torchtpu`** and **`sglang-torchtpu`**, our startup scripts in **[`scripts/patches/`](scripts/patches/README.md)** (`apply_vllm_torchtpu_bf16_patches.py` and `apply_sglang_bf16_patches.py`) fix two small upstream model-loader omissions at container startup:
  1. **TPU MXU `128 × 128` Tile Alignment (`hidden_size = 2880 -> 2944`)**:
     - TPU MXUs and Pallas grouped-matmul (`gmm_v2`) kernels require matrix dimensions to be exact multiples of **`128`**.
     - Most LLMs have a `hidden_size` divisible by `128` (e.g., `4096`, `5120`), whereas **`GPT-OSS 120B` has `hidden_size = 2880`** (`22.5 × 128`, leaving an unaligned remainder of `64` columns).
     - In `vllm-torchtpu`'s unquantized BF16 MoE layer (`VllmUnquantizedFusedMoEMethod`), upstream padded `intermediate_size` to a multiple of `128`, **but forgot to pad `hidden_size` (`2880 -> 2944`, adding `64` zero columns)**:
       ```python
       pad_H = (128 - (H % 128)) % 128  # 64 columns (pads 2880 -> 2944 = 23 * 128)
       if pad_H > 0:
           w13_weight = torch.nn.functional.pad(w13_weight, (0, pad_H))       # [16, 5760, 2880] -> [16, 5760, 2944]
           w2_weight  = torch.nn.functional.pad(w2_weight, (0, 0, 0, pad_H))  # [16, 2880, 2880] -> [16, 2944, 2880]
           if w2_bias is not None:
               w2_bias = torch.nn.functional.pad(w2_bias, (0, pad_H))         # [16, 2880]       -> [16, 2944]
       ```
     - Without that 6-line load-time padding, Pallas `gmm_v2` reads 64 uninitialized memory lanes past each row (`18.8 ms` TPOT). With [`apply_vllm_torchtpu_bf16_patches.py`](scripts/patches/apply_vllm_torchtpu_bf16_patches.py) applied before server launch, **`vllm-torchtpu` decode latency drops `3.2x` from `18.8 ms` down to `5.30–5.95 ms` TPOT (`2,284.8` out tok/s on `1k/1k`)**!
  2. **Standard Hugging Face BF16 Tensor Mapping & `swigluoai` Deinterleaving**:
     - Because OpenAI originally released `gpt-oss-120b` only in MXFP4, upstream `gpt_oss.py` in `sglang-torchtpu` and `vllm-torchtpu` assumed MXFP4 tensor names (`gate_up_proj_blocks`).
     - [`apply_sglang_bf16_patches.py`](scripts/patches/apply_sglang_bf16_patches.py) maps standard Hugging Face BF16 expert tensor names (`gate_up_proj`, `down_proj`), deinterleaves OpenAI's interleaved `swigluoai` gate/up channels (`[::2]` and `[1::2]`), and pads `2880 -> 2944` once during weight loading — enabling **`sglang-torchtpu` to achieve the #1 lowest latency on Native BF16 (`739.5 ms` TTFT on `22k` input, `4.97 ms` TPOT, `2,530.8` out tok/s on 1 host, and `5,084.5` out tok/s = `2.01x` linear scaling on 2 hosts / 8 chips)**.

---

## 3. Prerequisites & Variables (Set Once in Your Terminal)

Install the [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) and `kubectl` (`gcloud components install kubectl gke-gcloud-auth-plugin`), or open **Google Cloud Shell** in the Console. Then set your environment variables:

```bash
export PROJECT_ID="<YOUR_GCP_PROJECT_ID>"
export REGION="us-central1"                     # Region where your DWS Calendar reservation lives
export ZONE="us-central1-c"                     # Specific zone of your DWS Calendar reservation
export RESERVATION_NAME="<YOUR_DWS_CALENDAR_RESERVATION_NAME>"

export CLUSTER_NAME="tpu-v7x-bench-cluster"
export VPC_NAME="tpu-v7x-vpc"
export SUBNET_NAME="tpu-v7x-subnet"
export GCS_BUCKET="${PROJECT_ID}-tpu-v7x-models"
export HF_TOKEN="<YOUR_HUGGINGFACE_READ_TOKEN>" # Must have access to openai/gpt-oss-120b & lmsys/gpt-oss-120b-bf16

gcloud auth login
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"
gcloud config set compute/zone "${ZONE}"
```

> **One-Command Automated GKE + DWS Calendar Provisioning**:
> You can run [`k8s/provision_gke_dws_calendar.sh`](k8s/provision_gke_dws_calendar.sh) (`NUM_HOSTS=1`, `2`, or `4`) to execute Steps 4 through 7 automatically, or follow the step-by-step commands below.

---

## 4. Enable Required Google Cloud APIs, VPC, Cloud NAT & GCS Bucket

```bash
# 1. Enable Compute, GKE, TPU, Storage, and Artifact Registry APIs
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  tpu.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com

# 2. Create custom VPC & Subnet with Private Google Access (for high-speed GCS weight streaming)
gcloud compute networks create "${VPC_NAME}" --subnet-mode=custom

gcloud compute networks subnets create "${SUBNET_NAME}" \
  --network="${VPC_NAME}" \
  --region="${REGION}" \
  --range="10.128.0.0/16" \
  --enable-private-ip-google-access

# 3. Create Cloud Router & Cloud NAT (allows private GKE nodes to pull container images & Hugging Face weights)
gcloud compute routers create "${VPC_NAME}-router" \
  --network="${VPC_NAME}" \
  --region="${REGION}"

gcloud compute routers nats create "${VPC_NAME}-nat" \
  --router="${VPC_NAME}-router" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges

# 4. Create regional GCS bucket for model weights & persistent XLA compilation cache
gcloud storage buckets create "gs://${GCS_BUCKET}" \
  --location="${REGION}" \
  --uniform-bucket-level-access
```

---

## 5. Verify Your Active DWS Calendar Mode Reservation

When your DWS Calendar reservation window starts, Compute Engine creates an active specific reservation in your zone. Verify its exact name, machine type (`tpu7x-standard-4t`), and available host count (`count` vs `inUseCount`):

```bash
gcloud compute reservations list \
  --filter="zone:(${ZONE})" \
  --format="table(name,status,zone,specificReservation.count,specificReservation.inUseCount,specificReservation.instanceProperties.machineType)"

# Set RESERVATION_NAME to the exact name shown above:
export RESERVATION_NAME="<YOUR_DWS_CALENDAR_RESERVATION_NAME>"
```

- `count: 1` -> **1 Host (`4` physical `v7x` chips, `2x2x1`)**
- `count: 2` -> **2 Hosts (`8` physical `v7x` chips, `2x2x2`)**
- `count: 4` -> **4 Hosts (`16` physical `v7x` chips, `2x2x4`)**

---

## 6. Create the Standard GKE Cluster & Bind Your DWS Calendar TPU v7x Node Pool

### Step 6.1: Create the GKE Cluster (with GCSFuse CSI Driver & KubeRay Operator Enabled)

```bash
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

gcloud container clusters get-credentials "${CLUSTER_NAME}" --region="${REGION}"

# Grant node service account access to your GCS model bucket
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

### Step 6.2: Create the Ironwood TPU (`tpu7x-standard-4t`) Node Pool Consuming Your DWS Calendar Reservation

Choose **Option A (1 Host = 4 Chips)**, **Option B (2 Hosts = 8 Chips)**, or **Option C (4 Hosts = 16 Chips)** matching your reservation size:

#### Option A: Single-Host (`1` Host = `4` Physical `v7x` Chips, `2x2x1` Topology)
```bash
gcloud container node-pools create tpu-v7x-1host-pool \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=1 \
  --reservation-affinity=specific \
  --reservation="${RESERVATION_NAME}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

#### Option B: 2-Host (`2` Hosts = `8` Physical `v7x` Chips, `2x2x2` Topology)
Per Google Cloud's [Reserve TPUs in Calendar mode on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-calendar-mode) guide, multi-host Ironwood node pools require a regional **`HIGH_THROUGHPUT` Workload Policy** specifying the `2x2x2` accelerator topology:
```bash
gcloud compute resource-policies create workload-policy tpu7x-2x2x2-policy \
  --region="${REGION}" \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=2x2x2

gcloud container node-pools create tpu-v7x-2host-pool \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=2 \
  --placement-policy=tpu7x-2x2x2-policy \
  --reservation-affinity=specific \
  --reservation="${RESERVATION_NAME}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

#### Option C: 4-Host (`4` Hosts = `16` Physical `v7x` Chips, `2x2x4` Topology)
```bash
gcloud compute resource-policies create workload-policy tpu7x-2x2x4-policy \
  --region="${REGION}" \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=2x2x4

gcloud container node-pools create tpu-v7x-4host-pool \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=4 \
  --placement-policy=tpu7x-2x2x4-policy \
  --reservation-affinity=specific \
  --reservation="${RESERVATION_NAME}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

### Step 6.3: Create a Dedicated CPU Benchmark Client Node Pool (`c2-standard-16`)
Running `vllm bench serve` on a dedicated CPU node pool prevents client tokenization and async HTTP load generation from stealing host CPU cores from the TPU runtime:
```bash
gcloud container node-pools create cpu-bench-pool \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=c2-standard-16 \
  --num-nodes=1 \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

### Step 6.4: Verify TPU Node Pool Readiness
```bash
kubectl get nodes -L cloud.google.com/tpu-accelerator,cloud.google.com/tpu-topology
```
Each `tpu7x-standard-4t` node will report `google.com/tpu: 4` (`4` physical Ironwood chips = `8` TensorCores).

---

## 7. Stage `GPT-OSS 120B` Checkpoints to Your GCS Bucket (Run Once)

Stage both the **Pure Unquantized BF16** (`lmsys/gpt-oss-120b-bf16`, `233.7 GB`) and **Native MXFP4** (`openai/gpt-oss-120b`, `65.3 GB`) checkpoints into your regional GCS bucket using [`scripts/stage_hf_to_gcs.py`](scripts/stage_hf_to_gcs.py):

```bash
pip install -U huggingface_hub google-cloud-storage

# 1. Stage Pure Unquantized BF16 checkpoint (lmsys/gpt-oss-120b-bf16, ~234 GB)
python3 scripts/stage_hf_to_gcs.py \
  --repo-id lmsys/gpt-oss-120b-bf16 \
  --gcs-bucket "${GCS_BUCKET}" \
  --gcs-prefix models/lmsys/gpt-oss-120b-bf16

# 2. Stage Native MXFP4 checkpoint (openai/gpt-oss-120b, ~65.3 GB)
python3 scripts/stage_hf_to_gcs.py \
  --repo-id openai/gpt-oss-120b \
  --gcs-bucket "${GCS_BUCKET}" \
  --gcs-prefix models/openai/gpt-oss-120b
```

---

## 8. Deploy & Benchmark `GPT-OSS 120B` on GKE (Native BF16 & MXFP4 Across `sglang-torchtpu`, `vllm-torchtpu`, and `vllm-tpu`)

### 8.1 Deploy Serving Pods on GKE (`1`, `2`, or `4` Hosts)

All Kubernetes manifests in [`k8s/`](k8s/) mount your GCS bucket via `gke-gcsfuse-volumes`, configure `/dev/shm`, and automatically apply the startup patches (`scripts/patches/`) when needed. Simply substitute `<YOUR_GCS_BUCKET>` with `${GCS_BUCKET}`:

#### Option 1: Pure Unquantized BF16 (`lmsys/gpt-oss-120b-bf16`, `234 GB`) on `sglang-torchtpu` (`TP=8, EP=8` per host — Lowest Latency: `739.5 ms` TTFT, `4.97 ms` TPOT)
```bash
sed "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" k8s/gpt-oss-120b-bf16-sglang-v7x.yaml | kubectl apply -f -

# For 2 hosts (8 chips -> 5,084.5 out tok/s = 2.01x scaling) or 4 hosts (16 chips), scale replicas:
kubectl scale deployment/gpt-oss-120b-bf16-sglang-v7x --replicas=2
```

#### Option 2: Pure Unquantized BF16 (`lmsys/gpt-oss-120b-bf16`, `234 GB`) on `vllm-tpu` (JAX `TP=8, --enable-expert-parallel` per host — Zero Patches: `2,540.3` out tok/s)
```bash
sed "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" k8s/gpt-oss-120b-bf16-vllm-tpu-v7x.yaml | kubectl apply -f -

# Scale to 2 hosts (8 chips) or 4 hosts (16 chips):
kubectl scale deployment/gpt-oss-120b-bf16-vllm-tpu-v7x --replicas=2
```

#### Option 3: Native MXFP4 (`openai/gpt-oss-120b`, `65.3 GB`) on `vllm-tpu` (`TP=2, DP=4` per host — Peak Throughput: `5,847` tok/s on 1 host, `14,277` on 2 hosts, `20,367` on 4 hosts)
```bash
sed "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" k8s/gpt-oss-120b-vllm-tpu-v7x.yaml | kubectl apply -f -

# Scale to 2 hosts (8 chips) or 4 hosts (16 chips):
kubectl scale deployment/gpt-oss-120b-vllm-tpu-v7x --replicas=2
```

#### Option 4: Native MXFP4 (`openai/gpt-oss-120b`, `65.3 GB`) on `sglang-torchtpu` (`TP=8, EP=8` + Optional `EAGLE3` Speculative Decoding — `4.09 ms` TPOT)
```bash
sed "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" k8s/gpt-oss-120b-sglang-eagle3-v7x.yaml | kubectl apply -f -
```

### 8.2 Expose a Kubernetes `ClusterIP` Service & Run Benchmarks from the CPU Node Pool

```bash
# Expose your active deployment as a ClusterIP Service on port 8000
kubectl expose deployment gpt-oss-120b-bf16-sglang-v7x \
  --name=gpt-oss-120b-svc \
  --port=8000 \
  --target-port=8000 \
  --type=ClusterIP

# Watch server logs until warmup & XLA compilation finish
kubectl logs -f deploy/gpt-oss-120b-bf16-sglang-v7x
```

Once `/v1/models` returns HTTP 200, run the full reproducible benchmark suite (`configs/bf16/*.json` or `configs/*.json`) or `vllm bench serve` from a pod scheduled on `cpu-bench-pool`:

```bash
kubectl run bench-client --rm -it --restart=Never \
  --image=vllm/vllm-tpu:latest \
  --overrides='{"spec":{"nodeSelector":{"cloud.google.com/gke-nodepool":"cpu-bench-pool"}}}' \
  -- bash -c "
    vllm bench serve \
      --backend vllm \
      --base-url http://gpt-oss-120b-svc.default.svc.cluster.local:8000 \
      --model openai/gpt-oss-120b \
      --tokenizer lmsys/gpt-oss-120b-bf16 \
      --dataset-name random \
      --random-input-len 1024 \
      --random-output-len 1024 \
      --max-concurrency 64 \
      --num-prompts 256 \
      --ignore-eos \
      --trust-remote-code
  "
```

> **Tip for Multi-Host Long-Context (`22k / 4k`) Benchmarking**:
> When scaling across 2 or 4 hosts (`--replicas=2` or `4`), you can also run [`scripts/lb_proxy.py`](scripts/lb_proxy.py) (`python3 scripts/lb_proxy.py --port 8000 --backends http://<POD_0_IP>:8000 http://<POD_1_IP>:8000`) inside the benchmark pod for active least-connections (`least_conn`) routing across long-context (`22k/4k`) requests.

---

## Appendix A: Optional Single-Host (`1` Host = `4` Chips) Standalone GCE VM Testing

If you only want to test a **single host (`1 × tpu7x-standard-4t` = `4` chips)** without creating a GKE cluster, you can create a standalone GCE VM consuming 1 host from your DWS Calendar reservation:

```bash
gcloud compute instances create tpu-v7x-single-host \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --network-interface="network=${VPC_NAME},subnet=${SUBNET_NAME},no-address" \
  --reservation-affinity=specific \
  --reservation="${RESERVATION_NAME}" \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=500GB \
  --boot-disk-type=hyperdisk-balanced \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

Then SSH into `tpu-v7x-single-host` (`gcloud compute ssh tpu-v7x-single-host --zone="${ZONE}" --tunnel-through-iap`) and use the single-host helper script [`gce/launch_gce_tpu_v7x_gpt_oss_120b.sh`](gce/launch_gce_tpu_v7x_gpt_oss_120b.sh) (`serve-bf16-sglang`, `serve-bf16-torchtpu`, `serve-bf16-vllm-tpu`, `serve-mxfp4-vllm-tpu`, or `serve-mxfp4-sglang`). For any **2-host (`8`-chip) or 4-host (`16`-chip)** deployment, always use the **GKE workflow in Sections 6–8**.
