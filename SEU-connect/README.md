# seu-connect
登录东南大学校园网 (seu-wlan) 的脚本

## 使用前说明
使用前需要对seu-connect.sh脚本中的某些内容进行修改

第12行的 **sunriseUsername=USERNAME**，其中的USERNAME改成你本人的一卡通号

第18行的 **username=USERNAME**，其中的USERNAME改成你本人的一卡通号

第18行的 **password=PASSWORD**，其中的PASSWORD改成你的密码(需要经过base64编码)

base64编码方式如下，在命令行中输入
 ```
 echo "PASS" | base64
 ```
 其中PASS为你的密码，用输出结果替换第18行的**PASSWORD**


## 使用方法
在命令行中切换到当前目录下，然后输入以下命令并回车
```
./seu-connect.sh
```

## TODO
- 自动连接ESSID: seu-wlan，而无需手动连接
- 将脚本命令设置为默认路径
