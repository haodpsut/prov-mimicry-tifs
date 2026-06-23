#!/usr/bin/env bash
# One-time setup on the GPU server: clone public MAGIC, unzip its shipped graphs.
set -e

if [ ! -d MAGIC ]; then
  git clone https://github.com/FDUDSDE/MAGIC.git
fi

cd MAGIC
for d in streamspot wget; do
  if [ ! -f "data/$d/graphs.pkl" ] && [ -f "data/$d/graphs.zip" ]; then
    echo "unzipping data/$d/graphs.zip ..."
    unzip -o "data/$d/graphs.zip" -d "data/$d/"
  fi
done
cd ..

echo "checkpoints:"; ls -la MAGIC/checkpoints || true
echo
echo "Setup done. Now run, e.g.:"
echo "  python smoke_reproduce.py --magic_root ./MAGIC --dataset streamspot --device 0"
echo "  python smoke_attack.py   --magic_root ./MAGIC --dataset streamspot --device 0"
