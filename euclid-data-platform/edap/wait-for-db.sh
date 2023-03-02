# wait for database ready
set -e

db_host="$1"
db_port="$2"
db_password="$3"
shift 3
cmd="$@"


python /wait-for-db.py -h $db_host -P $db_port -p $db_password 


exec $cmd