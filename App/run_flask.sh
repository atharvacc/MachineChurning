export FLASK_APP=hello.py
export FLASK_ENV=development
if [[ "$1" == "local" ]]
	then
		flask run
else
	flask run --host=0.0.0.0 --port=8888
fi
