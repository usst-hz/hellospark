
import org.apache.spark.{SparkConf,SparkContext}


/**
  * Created by hz on 2017/5/14.
  */
object corfliter {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("cf user-based").setMaster("local")
    val sc = new SparkContext(conf)

    //提取(userid,itemed,rating) from ratings data
    val oriRatings = sc.textFile("E://ratings1.dat").map(line =>{
      val fields=line.split("::")
      (fields(0).toLong,fields(1).toLong,fields(2).toInt)
    })
    //oriRatings: (Long,Long,Int)

    //先对用户进行聚类，过滤每个用户至多评价100部电影
    val ratings = oriRatings.groupBy(k=>k._1).flatMap(x=>(x._2.toList.sortWith((x,y)=>x._3>y._3).take(100)))

    //一个用户匹配他评价的物品
    val user2manyItem = ratings.groupBy(_._1)

    //一个用户匹配他评价物品的数量(user,num)
    val numPrefPerUser = user2manyItem.map(g =>(g._1,g._2.size))

    //把用户评价的数量与用户每个评价记录join起来,ratingWithSize:(user,item,rating,numPerfs)
    val ratingWithSize = user2manyItem.join(numPrefPerUser).flatMap(joined =>{
      joined._2._1.map(f =>(f._1,f._2,f._3,joined._2._2))
    })

    ////(user, item, rating, numPrefs) ==>(item,(user, item, rating, numPrefs))
    val ratings2 = ratingWithSize.keyBy(tup => tup._2)

    //ratingPairs format (t,iterator((u1,t,pref1,numpref1),(u2,t,pref2,numpref2))) and u1<u2,对相同的物品进行分组
    //并且过滤掉自身匹配
    val ratingPairs = ratings2.join(ratings2).filter(f =>f._2._1._1 < f._2._2._1)

    //tempVectorCalcs表示2个用户对同一件物品的评价差异以及其他数理统计
    val tempVectorCalcs = ratingPairs.map(data =>{
      val key = (data._2._1._1,data._2._2._1)//对相同物品评价的两个用户(user1,user2)
      val stats =
        (data._2._1._3*data._2._2._3, //pref1*pref2
         data._2._1._3,//pref1
         data._2._2._3,//pref2
         math.pow(data._2._1._3,2),//square of pref1
         math.pow(data._2._2._3,2),//square of pref2
         data._2._1._4, //numprefs1 of user 1
          data._2._2._4) //numprefs2 of user 2
      (key,stats)
    })

    //对tempVectorCalcs进行用户组(user1,user2)分组,user1<user2
    val vectorCalcs = tempVectorCalcs.groupByKey().map(data =>{
      val key = data._1
      val vals= data._2
      val size =vals.size //(user1,user2)共同评价过的物品数量
      val dotProduct = vals.map(f =>f._1).sum //pref1*pref2的和
      val rating1Sum = vals.map(f =>f._2).sum //共同评价物品中 user1评分的中和
      val rating2Sum = vals.map(f=>f._3).sum
      val rating1SeqSum = vals.map(f=>f._4).sum
      val rating2SeqSum = vals.map(f=>f._5).sum
      val numPref1 = vals.map(f=>f._6).max
      val numPref2 = vals.map(f=>f._7).max
      (key,(size,dotProduct,rating1Sum,rating2Sum,rating1SeqSum,rating2SeqSum,numPref1,numPref2))
    }
    )

    //due to matrix is not symmetry(对称) , use half matrix build full matrix
    val inverseVectorCalcs = vectorCalcs.map(x=>((x._1._2,x._1._1),(x._2._1,x._2._2,x._2._4,x._2._3,x._2._6,x._2._5,x._2._8,x._2._7)))
    val vectorCalcsTotal = vectorCalcs ++ inverseVectorCalcs

    // compute similarity metrics for each movie pair,  similarities meaning user2 to user1 similarity
    val tempSimilarities =
      vectorCalcsTotal.map(fields => {
        val key = fields._1
        val (size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, numRaters, numRaters2) = fields._2
        val cosSim = cosineSimilarity(dotProduct, scala.math.sqrt(ratingNormSq), scala.math.sqrt(rating2NormSq))*
          size/(numRaters*math.log10(numRaters2+10))
        //val corr = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
        (key._1,(key._2, cosSim))
      })

    //similarities:针对user1取相似度排名前50的(user1,(user-n,sim1,n))
    val similarities = tempSimilarities.groupByKey().flatMap(x=>{
      x._2.map(temp=>(x._1,(temp._1,temp._2))).toList.sortWith((a,b)=>a._2._2>b._2._2).take(50)
    })

    val temp = similarities.filter(x=>x._2._2.equals(Double.PositiveInfinity))
    val similarTable = similarities.map(x=>(x._1,x._2._1,x._2._2))

    // ratings format (user,(item,raing))
    val ratingsInverse = ratings.map(rating=>(rating._1,(rating._2,rating._3)))

    //statistics format ((user2,item),(sim,sim*rating)),,,, ratingsInverse.join(similarities) fromating as (user,((item,rating),(user2,similar)))
    val statistics = ratingsInverse.join(similarities).map(x=>((x._2._2._1,x._2._1._1),(x._2._2._2,x._2._1._2*x._2._2._2)))

    // predictResult fromat ((user,item),predict)
    val predictResult = statistics.reduceByKey((x,y)=>((x._1+y._1),(x._2+y._2))).map(x=>(x._1,x._2._2/x._2._1))

    val filterItem = ratings.map(x=>((x._1,x._2),Double.NaN))
    val totalScore = predictResult ++ filterItem

    val finalResult = totalScore.reduceByKey(_+_).filter(x=> !(x._2 equals(Double.NaN))).map(x=>(x._1._1,x._1._2,x._2)).groupBy(x=>x._1).flatMap(x=>(x._2.toList.sortWith((x,y)=>x._3>y._3).take(10)))
//    finalResult.take(1)
    sc.stop()




  }

  def cosineSimilarity(dotProduct : Double, ratingNorm : Double, rating2Norm : Double) = {
    dotProduct / (ratingNorm * rating2Norm)
  }


}
