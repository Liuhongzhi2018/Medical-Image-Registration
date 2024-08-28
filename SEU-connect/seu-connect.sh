#!/bin/bash

echo "    Getting cookies from http://w.seu.edu.cn/......"

# get cookie from w.seu.edu.cn
cookie=`curl -I "http://w.seu.edu.cn/" 2> /dev/null | grep -o "PHPSESSID=.*;" | cut -d '=' -f 2 | cut -d ';' -f 1`

echo "    Cookie received: y"$cookie

echo "    Connecting seu-wlan......"

# set cookie, sunriseUsername=USERNAME(it should be your card number)
COOKIE="think_language=zh-CN; sunriseUsername=USERNAME; PHPSESSID="$cookie

# get reply json
# username=USERNAME(it should be your card number)
# password=PASSWORD(it should be your password encoded with base64)
reply_json=`curl -d "username=USERNAME&password=PASSWORD&enablemacauth=0" http://w.seu.edu.cn/index.php/index/login  --cookie $COOKIE 2> /dev/null`

# use python script to analyse json
python test_status.py $reply_json

# $? means the status number
if [ $? == "0" ]
then
    echo "    Status: 0"
    info=`echo $reply_json | cut -d ',' -f 2 | cut -d '"' -f 4 | cut -d '}' -f 1`
    echo "    "`python UTF8-CHN.py $info`

else
    echo "    Status: 1"
    info=`echo $reply_json | cut -d ',' -f 1 | cut -d '"' -f 4`
    echo "    "`python UTF8-CHN.py $info`
    ip=`echo $reply_json | cut -d ',' -f 5 | cut -d ':' -f 2 | cut -d '"' -f 2`
    echo "    ip: "$ip
fi
