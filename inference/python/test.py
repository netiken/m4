from torch.profiler import profile, record_function, ProfilerActivity
import torch

device = "cuda:3"
n_flows_total = torch.tensor(10, device=device)
n_flow_completed = torch.tensor(0, device=device)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    while n_flow_completed < n_flows_total / 2:
        n_flow_completed += torch.where(n_flow_completed < n_flows_total, 1, 0)
        n_flow_completed += torch.where(n_flow_completed < n_flows_total, 1, 0)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")


# while n_flow_completed < n_flows_total/2:
#     n_flow_completed += torch.where(n_flow_completed < n_flow_total, 1, 0)
#     n_flow_completed += torch.where(n_flow_completed < n_flow_total, 1, 0)
