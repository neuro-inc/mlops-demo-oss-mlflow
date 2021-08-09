import os
import sys
from locust import HttpUser, task, between


class Surname(HttpUser):
    wait_time = between(1, 2.5)

    @task
    def index(self):
        self.client.get("/", json={"line":"foo"})

    @task
    def test(self):
        self.client.get("/test")


if __name__ == '__main__':
    host = sys.argv[1] if len(sys.argv) > 1 else 'http://0.0.0.0:8080'
    os.system(f'locust -f rnn/test.py --headless -u 10 -r 50 --host {host}')
