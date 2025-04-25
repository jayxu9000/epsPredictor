from pyspark.sql import SparkSession
from pyspark.sql.functions import (unix_timestamp, expr, year, 
                                   quarter, lag, col, isnan, round, last, 
                                   explode, month, dayofmonth, coalesce,
                                   lit)
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window
spark = SparkSession.builder.appName("EPS_ETL").getOrCreate()

# EXTRACT

eps_input = "/opt/spark/files/in/eps_history.csv" 
income_statement_input = "/opt/spark/files/in/income_statement.csv" 
output_path = "/opt/spark/files/out/csv"  

eps_df = (spark.read.csv(eps_input, header=True, inferSchema=True)
            .withColumnRenamed("act_symbol","company_symbol")
            .withColumnRenamed("reported","reported_eps")
            .withColumnRenamed("estimate","estimated_eps"))
income_statement_df = (spark.read.csv(income_statement_input, header=True, inferSchema=True)
                 .withColumnRenamed("act_symbol","company_symbol"))

raw = (eps_df.join(income_statement_df, on=["company_symbol","date"], how="inner").drop("period"))



# TRANSFORM



w = Window.partitionBy("company_symbol").orderBy("date")
features_obs = (raw
    .withColumn("lag_eps",    lag("reported_eps",1).over(w))
    .withColumn("lag_netinc", lag("net_income",1).over(w))
    .dropna(subset=["lag_eps","lag_netinc"])
    .filter((col("lag_eps")!=0)&(col("lag_netinc")!=0))
    .withColumn("eps_growth",   expr("(reported_eps - lag_eps)    / lag_eps"))
    .withColumn("inc_growth",   expr("(net_income   - lag_netinc)  / lag_netinc"))
    .withColumn("eps_surprise", expr("(reported_eps - estimated_eps) / estimated_eps"))
    .filter(~isnan("eps_growth") & ~isnan("inc_growth") & ~isnan("eps_surprise"))
    .withColumn("date_ts", unix_timestamp("date","yyyy-MM-dd").cast("double"))
    .withColumn("year",    year("date"))
    .withColumn("quarter", quarter("date"))
)

indexer = StringIndexer(inputCol="company_symbol", outputCol="companyIdx", handleInvalid="keep")
encoder = OneHotEncoder(inputCol="companyIdx", outputCol="companyVec", dropLast=False)

featureCols = [
    "date_ts","estimated_eps","net_income", "sales", "average_shares", "income_from_continuing_operations",
    "lag_eps","lag_netinc","eps_growth","inc_growth","eps_surprise",
    "year","quarter","companyVec"
]
assembler = VectorAssembler(inputCols=featureCols, outputCol="features", handleInvalid="skip")

gbt = GBTRegressor(labelCol="reported_eps", featuresCol="features", maxIter=50, maxDepth=5, seed=42)

pipeline = Pipeline(stages=[indexer, encoder, assembler, gbt])

train, test = features_obs.randomSplit([0.8,0.2], seed=42)
model = pipeline.fit(train)

ranges = (raw.groupBy("company_symbol")
             .agg(
               expr("min(date)   as start"),
               expr("max(date)   as end"),
               expr("year(min(date)) as start_year"),
               expr("year(max(date)) as end_year")
             ))

md = (raw.select(
         "company_symbol",
         month("date").alias("month"),
         dayofmonth("date").alias("day")
     )
     .distinct()
)

ranges = ranges.withColumn("years", expr("sequence(start_year, end_year)"))
years  = ranges.select(
            "company_symbol", 
            explode("years").alias("year"),
            "start", "end"
         )

skeleton = (years
    .join(md, on="company_symbol")
    .withColumn("date", expr("make_date(year, month, day)"))
    .filter((col("date") >= col("start")) & (col("date") <= col("end")))
    .select("company_symbol", "date")
)


panel = skeleton.join(
    raw.select("company_symbol","date","reported_eps","estimated_eps","net_income", "sales","average_shares","income_from_continuing_operations"),
    on=["company_symbol","date"],
    how="left"
)

wff = Window.partitionBy("company_symbol").orderBy("date")\
             .rowsBetween(Window.unboundedPreceding, 0)
panel_ff = (panel
    .withColumn("estimated_eps",
        last("estimated_eps", ignorenulls=True).over(wff))
    .withColumn("net_income",
        last("net_income",    ignorenulls=True).over(wff))
    .withColumn("sales",
        last("sales",            ignorenulls=True).over(wff))
    .withColumn("average_shares",
        last("average_shares",   ignorenulls=True).over(wff))
    .withColumn("income_from_continuing_operations",
        last("income_from_continuing_operations",ignorenulls=True).over(wff))
    .withColumn("reported_eps_filled",
        last("reported_eps", ignorenulls=True).over(wff))
)

wl = Window.partitionBy("company_symbol").orderBy("date")
panel_feat = (
    panel_ff
    .withColumn("lag_eps",    coalesce(lag("reported_eps_filled",1).over(wl), col("reported_eps_filled")))
    .withColumn("lag_netinc", coalesce(lag("net_income",1).over(wl), col("net_income")))
    .withColumn("eps_growth",
        coalesce(expr("(reported_eps_filled - lag_eps)/lag_eps"), lit(0.0)))
    .withColumn("inc_growth",
        coalesce(expr("(net_income - lag_netinc)/lag_netinc"), lit(0.0)))
    .withColumn("eps_surprise",
        coalesce(expr("(reported_eps_filled - estimated_eps)/estimated_eps"), lit(0.0)))
    .withColumn("date_ts", unix_timestamp("date","yyyy-MM-dd").cast("double"))
    .withColumn("year",    year("date"))
    .withColumn("quarter", quarter("date"))
)

pred_all = model.transform(panel_feat)

result = (pred_all
    .withColumn("predicted_eps", round(col("prediction"),  2))
    .withColumn("difference",    round(col("predicted_eps") - col("reported_eps"), 2))
    .select("company_symbol","date",
            "reported_eps","estimated_eps","net_income",
            "sales","average_shares","income_from_continuing_operations","predicted_eps","difference")
)

panel_ff.write.parquet("/opt/spark/files/out/panel_ff.parquet", mode="overwrite")

model.write().overwrite().save("/opt/spark/files/out/eps_gbt_model")

# LOAD

jdbc_options = {
    "url": spark.conf.get("spark.mysql.epsPredictor.url"),
    "user": spark.conf.get("spark.mysql.epsPredictor.user"),
    "password": spark.conf.get("spark.mysql.epsPredictor.password"),
    "driver": "com.mysql.cj.jdbc.Driver",
}

result.write.format("jdbc").options(**jdbc_options, dbtable="eps_analysis").mode(
    "append"
).save()

spark.stop()