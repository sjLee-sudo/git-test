#!/bin/bash
IMAGE_FILE=/data/kistep_image/edap.tar
# 이미지 파일 확인
if [ ! -f "$IMAGE_FILE" ]; then
        >&2 echo "$IMAGE_FILE not exists, image tar file reqiured on the path"
        exit 1
fi
# 가동중인 분석엔진 컨테이너 중단 및 삭제
echo "stop and remove all analysis-engine"
podman ps -a | grep 'analysis-engine' | awk '{print $1}' | xargs --no-run-if-empty podman rm -f
# 기존 이미지 제거
echo "remove pre-builded image file"
podman rmi edap:latest
# 새로운 이미지 생성
echo "load new image file"
podman load -i $IMAGE_FILE
# 새로운 분석엔진 컨테이너 가동
echo "run new analysis-engine container"
sh /data/run_containers.sh
podman container ls
