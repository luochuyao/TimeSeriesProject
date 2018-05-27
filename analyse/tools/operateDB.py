from influxdb import InfluxDBClient
import numpy
import json
from . import InfluxDB
def op_db(filename):
    client = InfluxDBClient('localhost', 8086, 'root', '','ucr_2015')
    result1 = client.query('select * from '+filename+'_TRAIN;')
    file_content1 = numpy.array(InfluxDB.Source.turn_set_to_row(result1))
    result2 = client.query('select * from ' + filename + '_TEST;')
    file_content2 = numpy.array(InfluxDB.Source.turn_set_to_row(result2))
    return file_content1, file_content2