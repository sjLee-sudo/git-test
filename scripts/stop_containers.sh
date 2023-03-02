echo "stop and remove all analysis-engine"
podman ps -a | grep 'analysis-engine' | awk '{print $1}' | xargs --no-run-if-empty podman rm -f
