#-*- coding: utf8 -*-
#
# Author: Enrico Santus <esantus@mit.edu>
# Description: connector to structure the database

import pymysql.cursors
import json
import pandas as pd

connection = pymysql.connect(host='localhost',user='root',passwd='Lugon5556',db='EVALution2.0',charset='utf-8')
