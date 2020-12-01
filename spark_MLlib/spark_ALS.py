from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import Row,SparkSession
from pyspark.mllib.recommendation import Rating,ALS

def RDD_to_DF(rdd):
    RDD_table = rdd.map(lambda x:Row(user_id=int(x[0]),prpject_id=int(x[1]),evaluation=int(x[2]),datetime=x[3]))
    raw_DF = spark.createDataFrame(RDD_table)
    raw_DF.createOrReplaceTempView('evaluation')
    sqlDF = spark.sql(('SELECT * FROM evaluation WHERE evaluation <= 3'))
    sqlDF.show()

def Recommend_Inplicit(rdd,user_id,product_id):
    Ratings_RDD_Inplicit = rdd.map(lambda evaluation:(evaluation[0],evaluation[1],1))
    model_ALS_Inplicit = ALS.trainImplicit(ratings=Ratings_RDD_Inplicit,rank=10,iterations=10)
    if_recommend_movies_or_not = []
    for id in product_id:
        recommend_movies_i = model_ALS_Inplicit.predict(user=100,product=id)
        if_recommend_movies_or_not.append(recommend_movies_i)
    print('If recommend movies predicted by model_ALS:',if_recommend_movies_or_not)
    if_recommend_users_or_not = []
    for id in user_id:
        recommend_users_i = model_ALS_Inplicit.predict(user=id,product=200)
        if_recommend_users_or_not.append(recommend_users_i)
    print('If recommend users predicted by model_ALS:',if_recommend_users_or_not)

if __name__ == '__main__':
    conf = SparkConf().setAppName('ALS')
    sc = SparkContext(conf=conf)
    raw_data = sc.textFile('/usr/local/Cellar/apache-spark/3.0.0/libexec/sources/u.data')
    print(raw_data.count())
    print('The first data:')
    print(raw_data.first())

    print('---------------------------------')
    print('To point its significance:')
    data_preposed = raw_data.map(lambda x:x.split('\t'))
    spark = SparkSession \
            .builder \
            .appName('ALS_test') \
            .getOrCreate()
    RDD_to_DF(data_preposed)

    print('---------------------------------')
    # **Note that the method to transfer the RDD to DataFrame is indeed a efficient one to show its significance
    # However, there are other methods, more importantly
    # in order to apply the ALS, we need to transfer raw data by Rating
    print('Rating is a class inherited from namedtuple')
    Ratings_RDD = raw_data.map(lambda x:x.split('\t')[:3])
    print(Ratings_RDD.take(3))
    print('We need to transfer the list into tuple:')
    Ratings_tuple_RDD = Ratings_RDD.map(lambda list:(list[0],list[1],list[2]))
    print(Ratings_tuple_RDD.take(3))
    model_ALS = ALS.train(ratings=Ratings_tuple_RDD,rank=10,iterations=10)
    # As for user with user_id 100, recommend 5 movies
    print('Recommend 5 movies for user 100:')
    recommend_movies = model_ALS.recommendProducts(user=100,num=5)
    # print(type(recommend_result))
    # Show the evaluations of the recommended movies:
    n = 0
    for recommends in recommend_movies:
        n += 1
        evaluation = model_ALS.predict(user=int(recommends[0]),product=int(recommends[1]))
        print(f'The score for the recommend{n}:',evaluation)
    print('Moreover, we can also recommend 5 users recommending movie 200:')
    recommend_users = model_ALS.recommendUsers(product=200,num=5)
    m = 0
    for recommends in recommend_users:
        m += 1
        print(f'The score for the recommended user{m}:',recommends[2])

    print('---------------------------------')
    # Let us discern the construct of the Rating:
    # (user, product, rating)
    # Clearly, as for similar problems, it's no problem with the user and product
    # However, the rating may not be score,
    # sometimes we are only able to get the imformation about
    # whether the user is recommend on (interested in) the product,
    # i.e. only the data of boolean type.
    print('Let us see whether the two cases share similar recommend results:')
    # Here, we will write a general method in advance.
    Recommend_Inplicit(rdd=Ratings_RDD,user_id=[int(tp[0]) for tp in recommend_users],product_id=[int(tp[1]) for tp in recommend_movies])
    print('Unfortunately, we find out that the results are of great disparities')

    print('---------------------------------')
    # One may be puzzled that we have not use Rating
    print('How to use Rating:')
    Ratings_tuple_RDD_by_Rating = Ratings_RDD.map(lambda evaluation:Rating(evaluation[0],evaluation[1],evaluation[2]))
    print(Ratings_tuple_RDD_by_Rating.take(5))
    print('By the way, we recommend handling data by Rating!!!')