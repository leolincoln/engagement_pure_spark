#~/spark/bin/spark-submit --class "HitchCockProcess" --master "spark://wolf.iems.northwestern.edu:7077" target/TestCassandraMaven-0.0.1-SNAPSHOT-jar-with-dependencies.jar
#/opt/cloudera/parcels/CDH/bin/spark-submit \
if [ $# -lt 2 ]
    then
        echo "Usage: ./runYarn.bash fileName fileparameter"
        exit
fi
#if [ ! -f $1  ]; then
#    echo "file "$1" Not found!"
#    echo "Usage: ./runYarn.bash fileName"
#    exit
#fi
#--num-executors 4 \
#--master yarn \
export HADOOP_CONF_DIR=/etc/alternatives/hadoop-conf 
/opt/cloudera/parcels/CDH/bin/spark-submit \
--deploy-mode client \
--name $2 \
--executor-cores 6 \
--executor-memory 15g \
$1 $2
