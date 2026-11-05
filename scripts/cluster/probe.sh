# CLUSTER (login node), read-only: round 1 of docs/CLUSTER.md. Run from the laptop with
#   scripts/cluster/ssh.sh scripts/cluster/probe.sh
# Prints where we are, the OS and glibc, storage and quota, and what's already installed.
# Writes nothing.
set -u
echo "== host"
hostname
case "$(hostname)" in
  gh-login*) echo "OK: DeltaAI login node" ;;
  *) echo "WRONG CLUSTER (expected gh-loginNN): stop"; exit 1 ;;
esac
echo "== platform"
uname -m
grep PRETTY_NAME /etc/os-release
ldd --version | head -1
echo "== storage (HOME=$HOME)"
quota
ls -ld /projects/bhhv* /work/nvme/bhhv* /work/hdd/bhhv* 2>&1
echo "== allocation"
accounts 2>&1 | head -20
echo "== already present?"
ls -ld ~/gaitlab ~/.local/bin/uv ~/.cache/uv ~/.cache/huggingface 2>&1
echo "== containers"
command -v apptainer && apptainer --version
ls /sw/user/NGC_containers/ 2>&1 | grep -i pytorch
echo "== internet from the login node (HTTP status codes)"
for url in https://pypi.org/simple/ https://pypi.nvidia.com/ https://huggingface.co/api/models/google/gemma-3-270m-it; do
  printf '%s ' "$url"
  curl -sS -o /dev/null -w '%{http_code}\n' --max-time 10 "$url" 2>&1
done
