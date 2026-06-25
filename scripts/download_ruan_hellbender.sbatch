#!/bin/bash
#SBATCH --job-name=ruan_data
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ruan_data_%j.out
#SBATCH --error=logs/ruan_data_%j.err

set -euo pipefail

mkdir -p logs external
cd external

if [[ ! -d ruan_fdia/.git ]]; then
  git clone https://github.com/ruanjiaqi01/False-Data-Injection-Attack-Dataset ruan_fdia
else
  cd ruan_fdia
  git pull
  cd ..
fi

cd ruan_fdia

for BUS in 30 118; do
  cd "${BUS}-bus"
  mkdir -p extracted
  if [[ ! -f "${BUS}-bus-dataset-full.zip" ]]; then
    zip -s 0 "${BUS}-bus-dataset.zip" --out "${BUS}-bus-dataset-full.zip"
  fi
  unzip -o "${BUS}-bus-dataset-full.zip" -d extracted
  cd ..
done

echo "Ruan dataset ready at $(pwd)"
find . -maxdepth 3 -type f \( -name "*.mat" -o -name "*.zip" -o -name "*.z[0-9][0-9]" \) -printf "%p\t%s bytes\n" | sort
