gcloud compute ssh instance-20250214-072311
tmux ls
tmux detach -s training
tmux kill-session -t training
