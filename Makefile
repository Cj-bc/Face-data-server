PREFIX := /usr/local
DST_FILES := $(PREFIX)/share
DST_BIN := $(PREFIX)/bin

init:
	which pipenv >/dev/null 2>&1 || pip install pipenv
	test "$(DEV)" = "true" && pipenv install --dev || pipenv install

run:
	./face-data-server

test:
	pipenv run pytest

lint:
	pipenv run flake8 .
	pipenv run mypy tests faceDetection main.py

install: init
	test -d $(DST_FILES)/face-data-server || mkdir $(DST_FILES)/face-data-server
	cp -r face-data-server main.py Pipfile Pipfile.lock pytest.ini tests Makefile $(DST_FILES)/face-data-server
	test -d $(DST_FILES)/face-data-server/faceDetection || mkdir $(DST_FILES)/face-data-server/faceDetection
	cp faceDetection/*py $(DST_FILES)/face-data-server/faceDetection
	test -d $(DST_FILES)/face-data-server/faceDetection/learned-models || mkdir $(DST_FILES)/face-data-server/faceDetection/learned-models
	cp faceDetection/learned-models/helen-dataset.dat $(DST_FILES)/face-data-server/faceDetection/learned-models/
	cd $(DST_FILES)/face-data-server; ln -s "$(DST_FILES)/face-data-server/face-data-server" $(DST_BIN)/face-data-server; pipenv install

uninstall:
	unlink $(DST_BIN)/face-data-server
	rm -r $(DST_FILES)/face-data-server
