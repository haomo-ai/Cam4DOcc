echo "-------------"
echo "load config from local path:" $1
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

bash tools/dist_train.sh $config $2 ${@:3}