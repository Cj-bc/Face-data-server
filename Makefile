PREFIX := /usr/local
DST_FILES := $(PREFIX)/share
DST_BIN := $(PREFIX)/bin
SED_EXEC := $(if $(shell sed --version 2>/dev/null | grep "GNU"),sed -i, sed -i "")

init:
	which pipenv >/dev/null 2>&1 || pip install pipenv
	test "$(DEV)" = "true" && pipenv install --dev || pipenv install

run:
	./face-data-server

test:
	pipenv run pytest

lint:
	pipenv run flake8 .
	pipenv run mypy --strict tests FaceDataServer main.py

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


updateProto:
	pipenv run python -m grpc_tools.protoc -Iprotos --python_out=FaceDataServer/ \
		--grpc_python_out=FaceDataServer/ protos/faceDataServer.proto
		$(SED_EXEC) 's/import faceDataServer_pb2/import FaceDataServer.faceDataServer_pb2/' FaceDataServer/faceDataServer_pb2_grpc.py
