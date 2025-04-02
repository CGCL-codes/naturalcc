
keyword="/home/starmage/.vscode-server"

ps -ef | grep "$keyword" | grep -v grep | awk '{print $2}' | while read pid
do
    sudo kill -9 $pid
done

# ps -aux | grep '/data03/starmage/miniconda3/envs/uicoder/bin/python' | grep -v grep | awk '{print $2}' | xargs sudo kill -9
