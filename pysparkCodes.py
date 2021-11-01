# DATA420-21S2 Assignment 2
# Name：Xin Gao (43044879)
#####################################################################################
# Data Processing Q1
#####################################################################################
# The overview of the structure of the datasets
!hdfs dfs -ls -R /data/msd/

# The size of each dataset
!hdfs dfs -du -h /data/msd/*/*

# Count the number of rows
! hdfs dfs -cat /data/msd/main/summary/metadata.csv.gz | gunzip | wc -l #1000000
! hdfs dfs -cat /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/* | gunzip | wc -l #994622
! hdfs dfs -cat /data/msd/genre/msd-MAGD-genreAssignment.tsv | wc -l #422713
! hdfs dfs -cat /data/msd/tasteprofile/triplets.tsv/* | gunzip | wc -l #48373585

#####################################################################################
# Data Processing Q2 (a)
#####################################################################################

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
conf = sc.getConf()

N = int(conf.get("spark.executor.instances"))
M = int(conf.get("spark.executor.cores"))
partitions = 4 * N * M 

# Filter the Taste Profile dataset to remove the songs which were mismatched
# Load the taste profile mismatches.txt
mismatches = spark.read.text("hdfs:///data/msd/tasteprofile/mismatches/sid_mismatches.txt")
mismatches.show(10, 100)

# Extract the song id and track id from raw text
mismatches = mismatches.select(
    F.trim(mismatches.value.substr(9,18)).alias("SONG_ID").cast(StringType()),
    F.trim(mismatches.value.substr(27,19)).alias("TRACK_ID").cast(StringType())
    )
mismatches.show(10, False)
print(mismatches.count()) #19094

# Load the taste profile manually_accepted.txt
accepted = spark.read.text("hdfs:///data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt")
accepted.show(10, 100)

# Extract the song id and track id from raw text
accepted = accepted.filter(
    accepted.value.startswith("< ERROR: ") # filter out the lines not needed
    ).select(
    F.trim(accepted.value.substr(11,18)).alias("SONG_ID").cast(StringType()),
    F.trim(accepted.value.substr(29,19)).alias("TRACK_ID").cast(StringType())
    )
accepted.show(10, False)
print(accepted.count()) #488

# Remove the matches manually accepted from the mismatches
mismatches = mismatches.join(
    accepted, how = "left_anti", on = "TRACK_ID")
mismatches.cache()
print(mismatches.count()) #19093

# Load the taste profile triplet.tsv
schema_triplets = StructType([
    StructField("USER_ID", StringType()),
    StructField("SONG_ID", StringType()),
    StructField("PLAY_COUNT", IntegerType())
    ])

triplets = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .option("delimiter", "\t")
    .schema(schema_triplets)
    .load("hdfs:///data/msd/tasteprofile/triplets.tsv/*")
    .repartition(partitions) #repartition to speed up
)
triplets.show(10, False)
print(triplets.count()) #48373586

# Remove the mismatched songs from triplets dataset
triplets = triplets.join(mismatches, how = "anti", on = "SONG_ID")
triplets.cache()
print(triplets.count()) #45795111

#####################################################################################
# Data Processing Q2 (b)
#####################################################################################
# Create a mapping of data type
audio_attribute_type_mapping = {
  "NUMERIC": DoubleType(),
  "real": DoubleType(),
  "string": StringType(),
  "STRING": StringType()
}

# Create a list of prefixes
audio_dataset_names = [
  "msd-jmir-area-of-moments-all-v1.0",
  "msd-jmir-lpc-all-v1.0",
  "msd-jmir-methods-of-moments-all-v1.0",
  "msd-jmir-mfcc-all-v1.0",
  "msd-jmir-spectral-all-all-v1.0",
  "msd-jmir-spectral-derivatives-all-all-v1.0",
  "msd-marsyas-timbral-v1.0",
  "msd-mvd-v1.0",
  "msd-rh-v1.0",
  "msd-rp-v1.0",
  "msd-ssd-v1.0",
  "msd-trh-v1.0",
  "msd-tssd-v1.0"
]

# Load the audio attributes and see the distinct attributes
attributes = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .load("hdfs:///data/msd/audio/attributes/*")
)
attributes.select('_c1').dropDuplicates().show(10, False)

# Define a function to automate the creation of features dataframes
def features_df(file):
    attributes = (spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .load("hdfs:///data/msd/audio/attributes/" + file + ".attributes.csv"))

    attributes = attributes.rdd.map(lambda x: (x[0], audio_attribute_type_mapping[x[1]])).collect()

    schema = StructType([StructField(attribute[0], attribute[1], True) for attribute in attributes])

    data = (spark.read.format("com.databricks.spark.csv")
            .option("header", "true")
            .option("inferSchema", "false")
            .schema(schema)
            .load(f"hdfs:///data/msd/audio/features/" + file + ".csv")
            .repartition(partitions)
            )
           

    return data

# We can create a features dataframe by choosing an index of the name in the audio_dataset_names
# E.g. print out the first features dataframe
from pretty import SparkPretty
pretty = SparkPretty()
print(pretty(features_df(audio_dataset_names[1]).head()))


#####################################################################################
# Audio Similarity Q1 (a)
#####################################################################################
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# Pick the methods features dataset and load it
features_methods = (
    features_df("msd-jmir-methods-of-moments-all-v1.0")
    .withColumn("MSD_TRACKID", F.col("MSD_TRACKID").substr(2, 18)) # get rid of quotes
    )

features_methods.cache()
print(features_methods.count())

# Peek at the schema
features_methods.printSchema()

# Produce descriptive statistics for each feature column
statistics = (
    features_methods
    .describe()
    .toPandas()
    .set_index("summary")
    .rename_axis(None)
    .T
)
print(statistics)

# Define the assembler to create a column with all features
assembler = VectorAssembler(
    inputCols=[col for col in features_methods.columns if col.startswith("Method")], # leave out the SD_TRACKID column
    outputCol="Features"
)
features_cor = (
    assembler
    .transform(features_methods)
    .select(["Features", "MSD_TRACKID"])
    )
features_cor.cache()
print(features_cor.count()) #994615
features_cor.show(10, False)

# Calculate the correlations between features
correlations = Correlation.corr(features_cor, 'Features', 'pearson').collect()[0][0].toArray()
print(correlations)
for i in range(0, correlations.shape[0]):
    for j in range(i + 1, correlations.shape[1]):
        if correlations[i, j] > 0.8: # set the threshold to 0.8
            print((i, j))
#(1, 2) remove 2
#(2, 3)
#(3, 4) remove 3
#(6, 7) remove 7
#(8, 9) remove 9
# Drop the highly correlated columns
features_methods_r = (
    features_methods
    .drop(features_methods.columns[2], features_methods.columns[3], features_methods.columns[7], features_methods.columns[9])
    )
features_methods_r.printSchema()


#####################################################################################
# Audio Similarity Q1 (c)
#####################################################################################
# Load genre data
schema_genre = StructType([
    StructField('TRACK_ID', StringType()),
    StructField('GENRE', StringType()),
    ])

genre = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .option("delimiter", "\t")
    .schema(schema_genre)
    .load("/data/msd/genre/msd-MAGD-genreAssignment.tsv")
    .repartition(partitions)
    )
genre.show(10, False)
print(genre.count()) #422714

# Create the dataset of genres for the songs that were matched
genre_matched = genre.join(mismatches, on = "TRACK_ID", how = "left_anti")
genre_matched.cache()
print(genre_matched.count()) #415350

# Save the genre matched dataset and plot the distribution
genre_matched = genre_matched.repartition(1)
genre_matched.write.mode("overwrite").csv("hdfs:///user/xga37/outputs/msd/genre_mathced")
!hdfs dfs -copyToLocal hdfs:///user/xga37/outputs/msd/genre_mathced /users/home/xga37/msd

# The visualization is done outside of pyspark
#import matplotlib.pyplot as plt
#genre_counts = (
    #genre.groupBy("GENRE")
    #.count()
    #.orderBy(F.col("count").desc())
    #)

#axes = plt.axes()
#axes.bar(genre_counts(F.col("GENRE"), genre_counts(F.col("counts"))
#axes.set_title("Distribution of genres")
#axes.set_xlabel("Genre")
#axes.set_ylabel("Counts")
#axes.grid(True)
#plt.show()

#####################################################################################
# Audio Similarity Q1 (d)
#####################################################################################
# Merge the genres dataset and features dataset so that every song has a label
features_methods_labeled = (
    features_methods_r
    .withColumnRenamed("MSD_TRACKID", "TRACK_ID")
    .join(
        genre_matched,
        on = "TRACK_ID",
        how = "inner"
        )
    )
features_methods_labeled.cache()
features_methods_labeled.show(10, False)
print(features_methods_labeled.count()) #413289

#####################################################################################
# Audio Similarity Q2 (b)
#####################################################################################
# Convert the genre column into a binary column that represents if the song is "Electronic"
def is_electronic(genre):
    if genre == "Electronic":
        return 1
    else:
        return 0

is_electronic_udf = F.udf(is_electronic, IntegerType())

features_methods_electronic = (
    features_methods_labeled
    .withColumn("label", is_electronic_udf(F.col("GENRE")))
    .drop("TRACK_ID", "GENRE")
    )
features_methods_electronic.cache()
features_methods_electronic.show(10, False)

# Print the class balance of the binary label
def print_class_balance(data, name):
    N = data.count()
    counts = data.groupBy("label").count().toPandas()
    counts["ratio"] = counts["count"] / N
    print(name)
    print(N)
    print(counts)
    print("")
    
print_class_balance(features_methods_electronic, "features")

#features
#413289
#   label   count     ratio
#0      1   40026  0.096847
#1      0  373263  0.903153

#####################################################################################
# Audio Similarity Q2 (c)
#####################################################################################
# Split the dataset into training and test sets using window
from pyspark.sql.functions import *
from pyspark.sql.window import *

assembler = VectorAssembler(
    inputCols=[col for col in features_methods_electronic.columns if col.startswith("Method")], # leave out the label column
    outputCol="Features"
    )
features = (
    assembler
    .transform(features_methods_electronic)
    .select(["Features", "label"])
    )

# Scaling and centering
from pyspark.ml.feature import StandardScaler
stdScaler = StandardScaler(
    inputCol="Features", outputCol="scaledFeatures", withStd=True, withMean=True
    )
features = (
    stdScaler
    .fit(features)
    .transform(features)
    .select(["scaledFeatures", "label"])
    )

temp = (
    features
    .withColumn("id", monotonically_increasing_id())
    .withColumn("Random", rand())
    .withColumn(
        "Row",
        row_number()
        .over(
            Window
            .partitionBy("label")
            .orderBy("Random")
        )
    )
)
training = temp.where(
    ((col("label") == 0) & (col("Row") < 373263 * 0.8)) |
    ((col("label") == 1) & (col("Row") < 40026 * 0.8))
)
training.cache()

test = temp.join(training, on="id", how="left_anti")
test.cache()

training = training.drop("id", "Random", "Row")
test = test.drop("id", "Random", "Row")

print_class_balance(features, "features")
print_class_balance(training, "training")
print_class_balance(test, "test")

#features
#413289
#   label   count     ratio
#0      1   40026  0.096847
#1      0  373263  0.903153

#training
#330630
#   label   count     ratio
#0      1   32020  0.096845
#1      0  298610  0.903155

#test
#82659
#   label  count     ratio
#0      1   8006  0.096856
#1      0  74653  0.903144

# Downsampling
training_downsampled = (
    training
    .withColumn("Random", rand())
    .where((col("label") != 0) | ((col("label") == 0) & (col("Random") < 4 * 0.096847)))
)
training_downsampled.cache()

print_class_balance(training_downsampled, "training_downsampled")

#training_downsampled
#89796
#   label  count     ratio
#0      1  32020  0.356586
#1      0  57776  0.643414

# Upsampling
import numpy as np
ratio = 10
n = 15
p = ratio / n  # ratio < n such that probability < 1

def random_resample(x, n, p):
    # Can implement custom sampling logic per class,
    if x == 0:
        return [0]  # no sampling
    if x == 1:
        return list(range((np.sum(np.random.random(n) > p))))  # upsampling
    return []  # drop

random_resample_udf = udf(lambda x: random_resample(x, n, p), ArrayType(IntegerType()))

training_upsampled = (
    training
    .withColumn("Sample", random_resample_udf(col("label")))
    .select(
        col("scaledFeatures"),
        col("label"),
        explode(col("Sample")).alias("Sample")
    )
    .drop("Sample")
)

print_class_balance(training_upsampled, "training_upsampled")

#458327
#   label   count     ratio
#0      1  159780  0.348616
#1      0  298610  0.651522

#####################################################################################
# Audio Similarity Q2 Logistic Regression Model
#####################################################################################
# Train logistic regression model
import numpy as np
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def with_custom_prediction(predictions, threshold, probabilityCol="probability", customPredictionCol="customPrediction"):

    def apply_custom_threshold(probability, threshold):
        return int(probability[1] > threshold)

    apply_custom_threshold_udf = udf(lambda x: apply_custom_threshold(x, threshold), IntegerType())

    return predictions.withColumn(customPredictionCol, apply_custom_threshold_udf(col(probabilityCol)))


def print_binary_metrics(predictions, threshold=0.5, labelCol="label", predictionCol="prediction", rawPredictionCol="rawPrediction", probabilityCol="probability"):

    if threshold != 0.5:

        predictions = with_custom_prediction(predictions, threshold)
        predictionCol = "customPrediction"

    total = predictions.count()
    positive = predictions.filter((col(labelCol) == 1)).count()
    negative = predictions.filter((col(labelCol) == 0)).count()
    nP = predictions.filter((col(predictionCol) == 1)).count()
    nN = predictions.filter((col(predictionCol) == 0)).count()
    TP = predictions.filter((col(predictionCol) == 1) & (col(labelCol) == 1)).count()
    FP = predictions.filter((col(predictionCol) == 1) & (col(labelCol) == 0)).count()
    FN = predictions.filter((col(predictionCol) == 0) & (col(labelCol) == 1)).count()
    TN = predictions.filter((col(predictionCol) == 0) & (col(labelCol) == 0)).count()

    binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol=rawPredictionCol, labelCol=labelCol, metricName="areaUnderROC")
    auroc = binary_evaluator.evaluate(predictions)

    print('actual total:    {}'.format(total))
    print('actual positive: {}'.format(positive))
    print('actual negative: {}'.format(negative))
    print('threshold:       {}'.format(threshold))
    print('nP:              {}'.format(nP))
    print('nN:              {}'.format(nN))
    print('TP:              {}'.format(TP))
    print('FP:              {}'.format(FP))
    print('FN:              {}'.format(FN))
    print('TN:              {}'.format(TN))
    print('precision:       {}'.format(TP / (TP + FP)))
    print('recall:          {}'.format(TP / (TP + FN)))
    print('accuracy:        {}'.format((TP + TN) / total))
    print('auroc:           {}'.format(auroc))

lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='label')

# Train model on normal traning dataset
lr_model = lr.fit(training)
predictions_lr = lr_model.transform(test)
predictions_lr.cache()
print("8-2 sampling")
print_binary_metrics(predictions_lr)

# Train model on down sampled training dataset
lr_model_downsampled = lr.fit(training_downsampled)
predictions_lr_downsampled = lr_model_downsampled.transform(test)
predictions_lr_downsampled.cache()
print("Down sampling")
print_binary_metrics(predictions_lr_downsampled)

# Train model on up sampled training dataset
lr_model_upsampled = lr.fit(training_upsampled)
predictions_lr_upsampled = lr_model_upsampled.transform(test)
predictions_lr_upsampled.cache()
print("Up sampling")
print_binary_metrics(predictions_lr_upsampled)

# Try different threshold on up sampled dataset
print_binary_metrics(predictions_lr_upsampled, threshold=0.2)
print_binary_metrics(predictions_lr_upsampled, threshold=0.3)
print_binary_metrics(predictions_lr_upsampled, threshold=0.4)

#####################################################################################
# Audio Similarity Q2 Random Forest Model
#####################################################################################
# Train random forest model
ran = RandomForestClassifier(featuresCol='scaledFeatures', labelCol='label')
ran_model = ran.fit(training_upsampled)

predictions_ran = ran_model.transform(test)
predictions_ran.cache()
print_binary_metrics(predictions_ran)

#####################################################################################
# Audio Similarity Q2 Gradient-boosted Tree Model
#####################################################################################
# Train the GBT model
gbt = GBTClassifier(featuresCol="scaledFeatures",labelCol='label')
gbt_model = gbt.fit(training_upsampled)

predictions_gbt = gbt_model.transform(test)
predictions_gbt.cache()
print_binary_metrics(predictions_gbt)

#####################################################################################
# Audio Similarity Q3 (b)
#####################################################################################
# Tune the hyperparameters using cross-validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

gbt_paramGrid = (
    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [2, 4, 8])
    .addGrid(gbt.maxBins, [24, 32, 48])
    .addGrid(gbt.stepSize, [0.01, 0.1])
    .build()
    )

cv = CrossValidator(estimator=gbt,
                    estimatorParamMaps=gbt_paramGrid,
                    evaluator=BinaryClassificationEvaluator(),
                    numFolds=5)
cv_model = cv.fit(training_upsampled)
predictions_cv = cv_model.transform(test)

print_binary_metrics(predictions_cv)

#actual total:    82659
#actual positive: 8006
#actual negative: 74653
#threshold:       0.5
#nP:              13168
#nN:              69491
#TP:              3912
#FP:              9256
#FN:              4094
#TN:              65397
#precision:       0.2970838396111786
#recall:          0.48863352485635775
#accuracy:        0.8384930860523355
#auroc:           0.787974693667973

#####################################################################################
# Audio Similarity Q4 (b)
#####################################################################################
# Convert the genre column into an integer index
from pyspark.ml.feature import StringIndexer

label_stringIdx = StringIndexer(inputCol = "GENRE", outputCol = "label")
features_index_model = label_stringIdx.fit(features_methods_labeled)
features_methods_index = features_index_model.transform(features_methods_labeled)

# Create the features with index for training
assembler = VectorAssembler(
    inputCols=[col for col in features_methods_index.columns if col.startswith("Method")], outputCol="Features"
    )
features_multiclass = (
    assembler
    .transform(features_methods_index)
    .select(["Features", "label"])
    )

features_multiclass.groupBy("label").count().orderBy(F.col("count").desc()).show(50, False)

#####################################################################################
# Audio Similarity Q4 (c)
#####################################################################################
# Scaling and centering
from pyspark.ml.feature import StandardScaler
stdScaler = StandardScaler(
    inputCol="Features", outputCol="scaledFeatures", withStd=True, withMean=True
    )
features_multiclass = (
    stdScaler
    .fit(features_multiclass)
    .transform(features_multiclass)
    .select(["scaledFeatures", "label"])
    )

# Split the dataset into training and test sets
training_multiclass, test_multiclass =  features_multiclass.randomSplit([0.8, 0.2])
training_multiclass.cache()
test_multiclass.cache()

# Observation reweighting
training_multiclass_weighted = (
    training_multiclass
    .withColumn(
        "Weight",
        when(col("label") == 0, 0.3)
        .when(F.col("label").isin([8,9,10,11,12,13,14,15,16]), 5.0)
        .when(F.col("label").isin([17,18,19,20]), 10.0)
        .otherwise(1.0)
    )
)

weights = (
    training_multiclass_weighted
    .groupBy("label")
    .agg(
        collect_set(col("Weight")).alias("Weights")
    )
    .toPandas()
)
print(weights)

# GBT does not work for multiclass classification
#gbt = GBTClassifier(featuresCol="scaledFeatures",labelCol='label', weightCol="Weight")
#gbt_model_multiclass = gbt.fit(training_multiclass_weighted)

#predictions_gbt_multiclass = gbt_model_multiclass.transform(test_multiclass)
#predictions_gbt_multiclass.cache()
#print_binary_metrics(predictions_gbt)

# Choose logistic regression to do the multiclass classfication
lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='label', weightCol="Weight")
lr_model_multiclass = lr.fit(training_multiclass_weighted)
predictions_multiclass = lr_model_multiclass.transform(test_multiclass)
predictions_multiclass.cache()

# Print the metrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

for i in range(21):
    print("Label ", i, ": Precision by label ", \
evaluator.evaluate(predictions_multiclass, {evaluator.metricName: "precisionByLabel", evaluator.metricLabel: i}),\
"   Recall by label ", \
evaluator.evaluate(predictions_multiclass, {evaluator.metricName: "recallByLabel", evaluator.metricLabel: i}))

#####################################################################################
# Song recommendation Q1 (a)
#####################################################################################
# Unique songs
triplets.select(F.countDistinct("SONG_ID")).show() #378310

# Unique users
triplets.select(F.countDistinct("USER_ID")).show() #1019318

#####################################################################################
# Song recommendation Q1 (b)
#####################################################################################
# Most active user by play counts
triplets.sort(F.col("PLAY_COUNT").desc()).show() #995 might be an outlier

user_activity = (
    triplets.groupBy("USER_ID")
            .agg(F.count("PLAY_COUNT").alias("NUMBER_OF_SONGS"),
                 F.sum("PLAY_COUNT").alias("TOTAL_USER_PLAY_COUNTS"))
            .orderBy(F.col("NUMBER_OF_SONGS").desc())
    )
user_activity.cache()
user_activity.show(20, False)  # The most active user played 4316 songs


song_popularity = (
    triplets.groupBy("SONG_ID")
            .agg(F.count("USER_ID").alias("NUMBER_OF_USERS"),
                 F.sum("PLAY_COUNT").alias("TOTAL_SONG_PLAY_COUNTS"))
            .orderBy(F.col("NUMBER_OF_USERS").desc())
    )

song_popularity.cache()
song_popularity.show(20, False)

#####################################################################################
# Song recommendation Q1 (c)
#####################################################################################                                            
# Save the dataset and plot the distribution
user_activity_output = user_activity.repartition(1)
user_activity_output.write.mode("overwrite").csv("hdfs:///user/xga37/outputs/msd/user_activity")
!hdfs dfs -copyToLocal hdfs:///user/xga37/outputs/msd/user_activity /users/home/xga37/msd

song_popularity_output = song_popularity.repartition(1)
song_popularity_output.write.mode("overwrite").csv("hdfs:///user/xga37/outputs/msd/song_popularity")
!hdfs dfs -copyToLocal hdfs:///user/xga37/outputs/msd/song_popularity /users/home/xga37/msd

#####################################################################################
# Song recommendation Q1 (d)
#####################################################################################
statistics = (
    user_activity
    .select("NUMBER_OF_SONGS", "TOTAL_USER_PLAY_COUNTS")
    .describe()
    .toPandas()
    .set_index("summary")
    .rename_axis(None)
    .T
)
print(statistics)

#                          count                mean              stddev  min      max
#NUMBER_OF_SONGS         1019318   44.92720721109605    54.9111319974742    3     4316
#TOTAL_USER_PLAY_COUNTS  1019318  128.82423149596102  175.43956510305063  3.0  13074.0

print(user_activity.approxQuantile("NUMBER_OF_SONGS", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05))
# [3.0, 14.0, 26.0, 49.0, 4316.0]
print(user_activity.approxQuantile("TOTAL_USER_PLAY_COUNTS", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05))
# [3.0, 31.0, 60.0, 137.0, 13074.0]

statistics = (
    song_popularity
    .select("NUMBER_OF_USERS", "TOTAL_SONG_PLAY_COUNTS")
    .describe()
    .toPandas()
    .set_index("summary")
    .rename_axis(None)
    .T
)
print(statistics)

#                         count                mean             stddev  min       max
#NUMBER_OF_USERS         378310  121.05181200602681  748.6489783736968    1     90444
#TOTAL_SONG_PLAY_COUNTS  378310   347.1038513388491  2978.605348838225  1.0  726885.0

print(song_popularity.approxQuantile("NUMBER_OF_USERS", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05))
#[1.0, 4.0, 11.0, 45.0, 90444.0]

print(song_popularity.approxQuantile("TOTAL_SONG_PLAY_COUNTS", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05))
#[1.0, 8.0, 31.0, 134.0, 726885.0]

# Remove songs which have been played less than N times
# Remove users who have listened to fewer than M songs
N = 31
M = 26
song_reduced = song_popularity.filter(F.col("TOTAL_SONG_PLAY_COUNTS") > N)
print(song_reduced.count()/378310)
user_reduced = user_activity.filter(F.col("NUMBER_OF_SONGS") > M)                                  
print(user_reduced.count()/1019318)

triplets_reduced = (
    triplets.join(song_reduced, on = "SONG_ID", how = "inner")
            .join(user_reduced, on = "USER_ID", how = "inner")
    )

triplets_reduced.cache()
triplets_reduced.show()
triplets_reduced.count() #36602346


#####################################################################################
# Song recommendation Q1 (e)
#####################################################################################

from pyspark.ml.feature import StringIndexer

# Encoding

user_id_indexer = StringIndexer(inputCol="USER_ID", outputCol="user_id_encoded")
song_id_indexer = StringIndexer(inputCol="SONG_ID", outputCol="song_id_encoded")

user_id_indexer_model = user_id_indexer.fit(triplets_reduced)
song_id_indexer_model = song_id_indexer.fit(triplets_reduced)

triplets_reduced = user_id_indexer_model.transform(triplets_reduced)
triplets_reduced = song_id_indexer_model.transform(triplets_reduced)

# Split into the training and test datasets and make sure the all the test users are in training dataset

(training, test) = triplets_reduced.randomSplit([0.7, 0.3])

test_not_training = test.join(training, on="USER_ID", how="left_anti")

training.cache()
test.cache()
test_not_training.cache()

print(f"training:          {training.count()}")
print(f"test:              {test.count()}")
print(f"test_not_training: {test_not_training.count()}")

counts = test_not_training.groupBy("USER_ID").count().toPandas().set_index("USER_ID")["count"].to_dict()

temp = (
    test_not_training
    .withColumn("id", monotonically_increasing_id())
    .withColumn("random", rand())
    .withColumn(
        "row",
        row_number()
        .over(
            Window
            .partitionBy("USER_ID")
            .orderBy("random")
        )
    )
)

for k, v in counts.items():
    temp = temp.where((col("USER_ID") != k) | (col("row") < v * 0.7))

temp = temp.drop("id", "random", "row")
temp.cache()

temp.show(50, False)


training = training.union(temp.select(training.columns))
test = test.join(temp, on=["USER_ID", "song_id"], how="left_anti")
test_not_training = test.join(training, on="USER_ID", how="left_anti")

print(f"training:          {training.count()}")
print(f"test:              {test.count()}")
print(f"test_not_training: {test_not_training.count()}")

#training:          25618789
#test:              10983557
#test_not_training: 0


#####################################################################################
# Song recommendation Q2 (a)
#####################################################################################

from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics

# Use ALS to train an implicit matrix factorization model
als = ALS(maxIter=5, regParam=0.01, userCol="user_id_encoded", itemCol="song_id_encoded", ratingCol="PLAY_COUNT", implicitPrefs=True)
als_model = als.fit(training)
predictions = als_model.transform(test)

predictions = predictions.orderBy(col("USER_ID"), col("SONG_ID"), col("prediction").desc())
predictions.cache()

predictions.show(50, False)


#####################################################################################
# Song recommendation Q2 (b)
#####################################################################################

def extract_songs_top_k(x, k):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x][0:k]

extract_songs_top_k_udf = udf(lambda x: extract_songs_top_k(x, k), ArrayType(IntegerType()))

def extract_songs(x):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x]

extract_songs_udf = udf(lambda x: extract_songs(x), ArrayType(IntegerType()))

# Recommendations

k = 5

topK = als_model.recommendForAllUsers(k)

topK.cache()
print(topK.count()) #498505

topK.show(10, False)


recommended_songs = (
    topK
    .withColumn("recommended_songs", extract_songs_top_k_udf(col("recommendations")))
    .select("user_id_encoded", "recommended_songs")
)
recommended_songs.cache()
print(recommended_songs.count()) #498505

recommended_songs.show(10, 50)

# Relevant songs

relevant_songs = (
    test
    .select(
        col("user_id_encoded").cast(IntegerType()),
        col("song_id_encoded").cast(IntegerType()),
        col("PLAY_COUNT").cast(IntegerType())
    )
    .groupBy('user_id_encoded')
    .agg(
        collect_list(
            array(
                col("song_id_encoded"),
                col("PLAY_COUNT")
            )
        ).alias('relevance')
    )
    .withColumn("relevant_songs", extract_songs_udf(col("relevance")))
    .select("user_id_encoded", "relevant_songs")
)
relevant_songs.cache()
print(relevant_songs.count()) #498500

relevant_songs.show(10, 50)

# Choose users encoded id 333
recommended_songs.where(F.col("user_id_encoded") == 333).show(1, False)
#[12, 0, 87, 49, 93]


relevant_songs.where(F.col("user_id_encoded") == 333).show(1, False)
#[3956, 1447, 68178, 12267, 7984, 4363, 44421, 3230, 1881, 263, 5245, 1078, 20208, 3234, 4042, 2500, 9059,
#617, 26016, 9306, 399, 20807, 2080, 866, 2309, 35431, 89739, 1505, 2207, 4697, 465, 52429, 65859, 1658, 2973,
#21419, 1434, 7484, 20684, 47565, 65018, 36675, 57319, 12099, 53200, 1324, 46528, 37495, 10338, 30036, 6614, 2987,
#23599, 12329, 43810, 79156, 36722, 115779, 19310, 48145, 42988, 2028, 130949, 80913, 50299, 2335, 59725, 16222,
#27623, 33765, 28526, 68973, 21683, 568, 10058, 6440, 16190, 1383, 41603, 4431, 17337, 32650, 7291, 4203, 169038,
#27729, 79346, 72141, 37557, 18548, 38232, 5131, 3074, 9384, 745, 20956, 23765, 77028, 71556, 554, 43050, 1168,
#128956, 36947, 1706, 77072, 3854, 1106, 12015, 950, 1144, 47237, 8919, 100, 112676, 3053, 23773, 48771, 69786,
#12243, 238, 1112, 2826, 13356, 75782, 28582, 152753, 1492, 34228, 41483, 2918, 8394, 11211, 26801, 51909, 2891,
#9282, 12467, 11825, 2008, 35944, 546, 14842, 99462, 51630, 26813, 15420, 2193, 24885, 139457, 39369, 35814, 2219,
#3753, 124942, 6734, 2244, 8400, 39222, 11337]

# Choose an active user encoded id 3
recommended_songs.where(F.col("user_id_encoded") == 3).show(1, False)
relevant_songs.where(F.col("user_id_encoded") == 3).show(1, False)

#####################################################################################
# Song recommendation Q2 (c)
#####################################################################################
# Print the ranking metrics
combined = (
    recommended_songs.join(relevant_songs, on='user_id_encoded', how='inner')
    .rdd
    .map(lambda row: (row[1], row[2]))
)
combined.cache()
print(combined.count()) #498500

combined.take(1)

# ([107048, 127769, 129688, 113295, 145331], [1290, 923, 1178, 1542, 3512, 6852, 7060, 12253...])

rankingMetrics = RankingMetrics(combined)
k=10
print("Precision@10: ", rankingMetrics.precisionAt(k))
print("NDCG@10: ", rankingMetrics.ndcgAt(k))
print("Mean Average Precision (MAP): ", rankingMetrics.meanAveragePrecision)



