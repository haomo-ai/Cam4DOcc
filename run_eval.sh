echo "-------------"
echo "load config from local path:" $1
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

export PYTHONPATH="."

ckpt=$2
gpu=$3
bash tools/dist_test.sh $config $ckpt $gpu ${@:4}