DEPLOY_DIR=/mnt/e/kistep-deploy
docker rmi edap:latest
docker build . -t edap:latest
rm -rf $DEPLOY_DIR/edap.tar
docker save -o $DEPLOY_DIR/edap.tar edap:latest