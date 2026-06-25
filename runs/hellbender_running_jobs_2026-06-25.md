# Hellbender Running Job Configuration

Captured: 2026-06-25 local time.

These were active Open OnDemand Jupyter dashboard jobs for user `mkfqm`.

## Job 14655460

- Job name: `sys/dashboard/sys/hb_jupyter`
- State: `RUNNING`
- Account: `engineering`
- QOS: `normal`
- Partition: `gpu`
- Node: `g014`
- Batch host: `g014`
- Nodes: `1`
- Tasks: `1`
- CPUs per task: `16`
- Total CPUs: `16`
- Memory: `64G`
- GPU request: `gres:gpu:A100:1`
- TRES: `cpu=16,mem=64G,node=1,billing=16,gres/gpu=1`
- Time limit: `2-00:00:00`
- Submit time: `2026-06-24T05:49:51`
- Start time: `2026-06-24T09:40:08`
- End time: `2026-06-26T09:40:08`
- Work directory:
  `/home/mkfqm/ondemand/data/sys/dashboard/batch_connect/sys/hb_jupyter/output/f6ce0c1c-0859-4f63-a3fc-d35f5956a26e`
- Log file:
  `/home/mkfqm/ondemand/data/sys/dashboard/batch_connect/sys/hb_jupyter/output/f6ce0c1c-0859-4f63-a3fc-d35f5956a26e/output.log`

Equivalent Slurm shape:

```bash
sbatch \
  --partition=gpu \
  --account=engineering \
  --qos=normal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=64G \
  --gres=gpu:A100:1 \
  --time=2-00:00:00
```

## Job 14623875

- Job name: `sys/dashboard/sys/hb_jupyter`
- State: `RUNNING`
- Account: `engineering`
- QOS: `normal`
- Partition: `gpu`
- Node: `g012`
- Batch host: `g012`
- Nodes: `1`
- Tasks: `1`
- CPUs per task: `16`
- Total CPUs: `16`
- Memory: `64G`
- GPU request: `gres:gpu:A100:1`
- TRES: `cpu=16,mem=64G,node=1,billing=16,gres/gpu=1`
- Time limit: `2-00:00:00`
- Submit time: `2026-06-23T04:07:46`
- Start time: `2026-06-23T04:09:30`
- End time: `2026-06-25T04:09:32`
- Work directory:
  `/home/mkfqm/ondemand/data/sys/dashboard/batch_connect/sys/hb_jupyter/output/f3e9b323-ed39-4be7-bbae-1a11bf3e5e35`
- Log file:
  `/home/mkfqm/ondemand/data/sys/dashboard/batch_connect/sys/hb_jupyter/output/f3e9b323-ed39-4be7-bbae-1a11bf3e5e35/output.log`

Equivalent Slurm shape:

```bash
sbatch \
  --partition=gpu \
  --account=engineering \
  --qos=normal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=64G \
  --gres=gpu:A100:1 \
  --time=2-00:00:00
```

## Suggested CNN Detector Job Shape

For QFDIA CNN detector jobs, this should be plenty:

```bash
#SBATCH --partition=gpu
#SBATCH --account=engineering
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=08:00:00
```

