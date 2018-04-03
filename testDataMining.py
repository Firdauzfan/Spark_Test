# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:45:30 2018

@author: Firdauz_Fanani
"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, HiveContext
from pyspark.sql.functions import *
from pyspark.sql import Row
from pyspark.sql.types import *
from datetime import datetime, timedelta

#%%

sc = SparkContext()
sqlContext = SQLContext(sc)

#%%
train = sqlContext.read.format('csv').options(header='true', inferSchema='true').load('train.csv')
test = sqlContext.read.format('csv').options(header='true', inferSchema='true').load('test.csv')

#train = sqlContext.load(source="com.databricks.spark.csv", path = 'train.csv', header = True,inferSchema = True)
#test = sqlContext.load(source="com.databricks.spark.csv", path = 'PATH/test.csv', header = True,inferSchema = True)

train.printSchema()

train.head(10)

#train.count()

#%%

train.na.drop().count(),test.na.drop('any').count()

train = train.fillna(-1)
test = test.fillna(-1)

train.describe().show()

train.select('User_ID').show()

#%%

train.select('Product_ID').distinct().count(), test.select('Product_ID').distinct().count()

#%%

diff_cat_in_train_test=test.select('Product_ID').subtract(train.select('Product_ID'))
diff_cat_in_train_test.distinct().count()# For distict count

#%%

from pyspark.ml.feature import StringIndexer
plan_indexer = StringIndexer(inputCol = 'Product_ID', outputCol = 'product_ID1')
labeller = plan_indexer.fit(train)

#%%

Train1 = labeller.transform(train)
Test1 = labeller.transform(test)

Train1.show()

#%%

from pyspark.ml.feature import RFormula
formula = RFormula(formula="Purchase ~ Age+ Occupation +City_Category+Stay_In_Current_City_Years+Product_Category_1+Product_Category_2+ Gender",featuresCol="features",labelCol="label")

t1 = formula.fit(Train1)
#%%

train1 = t1.transform(Train1)
test1 = t1.transform(Test1)

train1.show()

train1.select('features').show()
train1.select('label').show()

#%%

from pyspark.ml.regression import RandomForestRegressor
rf = RandomForestRegressor()

(train_cv, test_cv) = train1.randomSplit([0.7, 0.3])
model1 = rf.fit(train_cv)
predictions = model1.transform(test_cv)

model1 = rf.fit(train_cv)
predictions = model1.transform(test_cv)


#%%

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator()
mse = evaluator.evaluate(predictions,{evaluator.metricName:"mse" })
import numpy as np
np.sqrt(mse), mse

model = rf.fit(train1)
predictions1 = model.transform(test1)

df = predictions1.selectExpr("User_ID as User_ID", "Product_ID as Product_ID", 'prediction as Purchase')

df.toPandas().to_csv('submission.csv')