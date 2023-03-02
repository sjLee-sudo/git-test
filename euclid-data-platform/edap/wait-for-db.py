import MySQLdb
import time
import sys
import getopt

host = ""
user = ""
password = ""
port = 3306
print("database checking start")
opts, args = getopt.getopt(sys.argv[1:], 'h:u:p:P:d:')

for opt, arg in opts:
	if opt in ("-h", "--db"):
		host = arg
	elif opt in ("-u", "--user"):
		user = arg
	elif opt in ("-p", "--password"):
		password = arg
	elif opt in ("-P", "--port"):
		port = int(arg)

message = """
	\n\n\n
	################################ 
	database connect:
	host = %s
	user = %s
	password = %s
	port = %s
	################################ \n\n\n
""" %  (host, user, password, port)

print(message)

while True:
	try:
		conn = MySQLdb.connect(host=host, user=user, passwd=password , port=port)

		while True:
			cursor = conn.cursor()
			cursor.execute("select 1")
			result = cursor.fetchone()

			if result and len(result) > 0:
				print("database connect successful")
				break
			else:
				print("database not connected... waiting...")
				time.sleep(1)

			cursor.close()

		conn.close()
		break
	except Exception as e:
		print("MYSQL not responds.. waiting for mysql up: %s" % e)
		time.sleep(1)
	