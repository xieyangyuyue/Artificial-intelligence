# 这个代码的目的是告诉大家, 原生pymysql很复杂, 大家掌握 pandas 读写mysql即可.
# 这个代码搭建看看就好了, 不用练.

# 导包
import pymysql

# 1. 获取连接对象(Python -> MySQL)        前台小姐姐
conn = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='123456',
    database='day01',
    charset='utf8'
)

# 2. 根据连接对象, 获取游标对象             我 -> 夯哥
cursor = conn.cursor()

# 3. 定义SQL语句
sql = 'select name, AKA from my_table limit 0,2'

# 4. 并执行SQL语句
cursor.execute(sql)

# 5. 获取结果集
result = cursor.fetchall()

# 6. 遍历结果集.
for row in result:
    print(row)

# 7. 释放资源.
cursor.close()
conn.close()