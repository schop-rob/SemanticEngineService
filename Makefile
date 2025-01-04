
.PHONY: all start-embedding start-search

all: start-embedding start-search

start-embedding:
	python embedding/serve.py &

start-search:
	python search/serve.py &