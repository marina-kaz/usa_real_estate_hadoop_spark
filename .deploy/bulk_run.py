import subprocess


UP_COMMANDS = [
'docker compose up -d',
'docker compose up -d --scale spark-yarn-worker=3'
]

LOG_PREFIXES = [
    '1node',
    '3nodes',
]

BASE_COMMAND = 'docker exec da-spark-yarn-master spark-submit --master yarn --deploy-mode cluster ./apps/client.py'

if __name__ == "__main__":
    for up_command, log_prefix in zip(UP_COMMANDS, LOG_PREFIXES):
        process = subprocess.run('docker compose down', shell=True)
        print(f'\nDown return: {process.returncode}')
        process = subprocess.run(up_command, shell=True)
        print(f'Up return: {process.returncode}\n')

        # for _ in range(3):
        #     command = BASE_COMMAND + f' --log-prefix {log_prefix}_optim --optimized'
        #     print(command)
        #     process = subprocess.run(command, shell=True)
        #     print('Return code:', process.returncode)

        for _ in range(10):
            command = BASE_COMMAND + f' --log-prefix {log_prefix}'
            print(command)
            process = subprocess.run(command, shell=True)
            print('Return code:', process.returncode)


