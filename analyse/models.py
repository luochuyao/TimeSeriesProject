# -*- coding:utf-8 -*-
from django.db import models

# Create your models here.

class spd_task(models.Model):

    task_id1 = models.CharField(max_length=128, verbose_name=u'任务序号')
    model_name1 = models.CharField(max_length=128, verbose_name=u'选择的模型')
    upload_model_name1 = models.CharField(max_length=128, verbose_name=u'上传的模型')
    file_name = models.CharField(max_length=128, verbose_name=u'选择的文件')
    upload_file_name = models.CharField(max_length=128, verbose_name=u'上传的文件')
    log_date = models.DateTimeField()

class Spd_data(models.Model):

    data_id1 = models.CharField(max_length=128,verbose_name=u'数据id')
    data_name = models.CharField(max_length=128,verbose_name=u'数据名字')

    def getDataAccordingName(self,filename):

        from sqlalchemy import create_engine
        import pandas as pd
        engine = create_engine('mysql+pymysql://root:root@localhost:3306/timeseries', echo=False)

        sql_train = 'SELECT * FROM ' + filename + '_TRAIN'
        train = pd.read_sql(sql_train,con=engine).values
        sql_test = 'SELECT * FROM ' + filename + '_TEST'
        test = pd.read_sql(sql_test,con=engine).values

        print('loading data successful')
        return train,test

    def writeData(self,data,filename):

        from sqlalchemy import create_engine

        import pandas as pd
        engine = create_engine('mysql+pymysql://root:root@localhost:3306/timeseries', echo=False)
        data.to_sql(filename + '_TRAIN', con=engine, index=False, if_exists='replace')
        sql = 'SELECT * FROM ' + filename
        engine.execute(sql).fetchall()

        return True


class UserCreateForm(models.Model):

    id = models.AutoField(primary_key=True)
    user_name = models.CharField(max_length=30)
    user_password = models.CharField(max_length=30)
    user_email = models.CharField(max_length=30)



