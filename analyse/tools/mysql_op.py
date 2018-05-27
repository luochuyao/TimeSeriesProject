import datetime
import MySQLdb
from MySQLdb.cursors import DictCursor

from TimeSeriesDataAnalysis.settings import DATABASES


class Db:

    def __init__(self):
        self.database = DATABASES['default']
        self.database_name = self.database['NAME']
        self.user = self.database['USER']
        self.password = self.database['PASSWORD']
        self.host = self.database['HOST']
        self.port = self.database['PORT']
        self.con = MySQLdb.connect(self.host, self.user, self.password, self.database_name, self.port, charset='utf8')
        self.con.autocommit(True)

    def close_connect(self):
        self.con.close()

    def get_tasksinfo(self):
        cur = self.con.cursor(DictCursor)
        # query_str = "select b.id as id,b.task_id as task_id,b.first as first,b.second as second,b.log_date as logdate,a.status as status,a.result as result,a.traceback as traceback " \
        #             "from celery_taskmeta a inner join test_celery_add b on a.task_id=b.task_id;"
        query_str = 'select b.id as id, b.task_id1 as task_id1, b.model_name1 as model_name1, b.upload_model_name1 as upload_model_name1,' \
                    'b.file_name as file name, b.upload_file_name as upload_file_name, b.log_date as log_date' \
                    'a.status as status,a.result as result,a.traceback as traceback ' \
                    'from celery_taskmeta a inner join test_celery_add b on a.task_id=b.task_id;'
        cur.execute(query_str)
        rows  = cur.fetchall()
        cur.close()
        return rows

