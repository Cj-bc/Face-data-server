PREFIX := /usr/local
DST_FILES := $(PREFIX)/share
DST_BIN := $(PREFIX)/bin

init:
	which pipenv >/dev/null 2>&1 || pip install pipenv
	test "$(DEV)" = "true" && pipenv install --dev || pipenv install

run:
	./face-data-server

# This will generate coverage information under docs/pytestCov
test:
	pipenv run pytest --cov-report=html --cov-report=term --cov=FaceDataServer

lint:
	pipenv run flake8 .
	pipenv run mypy --strict tests FaceDataServer main.py

install: init
	test -d $(DST_FILES)/face-data-server || mkdir $(DST_FILES)/face-data-server
	cp -r face-data-server main.py Pipfile Pipfile.lock pytest.ini tests Makefile FaceDataServer $(DST_FILES)/face-data-server
	test -d $(DST_FILES)/face-data-server/src || mkdir $(DST_FILES)/face-data-server/src
	cp src/helen-dataset.dat $(DST_FILES)/face-data-server/src
	cd $(DST_FILES)/face-data-server; ln -s "$(DST_FILES)/face-data-server/face-data-server" $(DST_BIN)/face-data-server; pipenv install

uninstall:
	unlink $(DST_BIN)/face-data-server
	rm -r $(DST_FILES)/face-data-server
