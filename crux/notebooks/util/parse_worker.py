import ast
from datetime import datetime
import pandas as pd


def parse_worker():
    start_time = datetime.strptime('2023/07/15 00:00', '%Y/%m/%d %H:%M')
    end_time = datetime.strptime('2023/07/31 23:59', '%Y/%m/%d %H:%M')

    df = pd.read_csv('../data/worker.csv')

    df.dropna(subset=['gmt_created', 'gmt_pod_finished'], how='any', inplace=True)
    df['gmt_created'] = df['gmt_created'].fillna('2023/07/15 00:00')
    df['gmt_pod_finished'] = df['gmt_pod_finished'].fillna('2023/07/31 23:59')
    df['gmt_created'] = df['gmt_created'].apply(
        lambda x:
        (datetime.strptime(x, '%Y/%m/%d %H:%M') - start_time).total_seconds() / 86400
    )
    df['gmt_pod_finished'] = df['gmt_pod_finished'].apply(
        lambda x:
        (datetime.strptime(x, '%Y/%m/%d %H:%M') - start_time).total_seconds() / 86400
    )
    df['duration'] = df['gmt_pod_finished'] - df['gmt_created']

    # drop jobs whose host_ip is not in the topo.csv
    host_ip_set = set(pd.read_csv('../data/topo.csv')['ip'])
    df = df[df['host_ip'].isin(host_ip_set)]

    res_nan_jobs = df[df['RES'].isna()]['job_name'].unique()
    # these jobs are cpu-only jobs (might be RL jobs?)
    for job in res_nan_jobs:
        # should only use a single worker
        assert len(df[df['job_name'] == job]) == 1
    df = df.dropna(subset=['RES'])

    df['RES'] = df['RES'].apply(ast.literal_eval)
    df['num_gpus'] = df['RES'].apply(lambda x: int(x['nvidia.com/gpu']) if 'nvidia.com/gpu' in x else None)
    # df['num_gpus'] = df['num_gpus'].astype(int)
    return df


df_worker_valid = parse_worker()
df_worker_GPU = df_worker_valid[df_worker_valid['num_gpus'].notna()]


def parse_job():
    global df_worker_GPU
    jobs = df_worker_GPU['job_name'].unique()
    groups = df_worker_GPU.groupby('job_name')

    list_start_time = []
    list_end_time = []
    list_num_gpus = []
    for job in jobs:
        start_time = groups.get_group(job)['gmt_created'].min()
        end_time = groups.get_group(job)['gmt_pod_finished'].max()
        num_gpus = groups.get_group(job)['num_gpus'].sum()

        list_start_time.append(start_time)
        list_end_time.append(end_time)
        list_num_gpus.append(num_gpus)

    df_jobs = pd.DataFrame({
        "job_name": jobs,
        "start_time": list_start_time,
        "end_time": list_end_time,
        "num_gpus": list_num_gpus
    })

    df_jobs['duration'] = df_jobs['end_time'] - df_jobs['start_time']
    return df_jobs


df_worker_jobs = parse_job()

