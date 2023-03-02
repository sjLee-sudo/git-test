from multiprocessing import cpu_count
workers = 5
worker_class = 'sync'
timeout = 50000
bind = '0.0.0.0:8000'
max_requests = 1024
max_requests_jitter = 50 
limit_request_line = 0 
limit_request_field_size = 0 
accesslog ='/app/logs/access.log'
errorlog = '/app/logs/error.log'
log_level='info'
capture_output=True
print(f'''
workers: {workers}
timeout: {timeout} 
accesslog: {accesslog}
errorlog: {errorlog}
worker-class: {worker_class}
bind: {bind} 
max-request: {max_requests}
max-request-jitter : {max_requests_jitter}
limit-request-line : {limit_request_line}
limit-request-field-size: {limit_request_field_size}
 ''')