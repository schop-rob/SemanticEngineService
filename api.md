# API Guide

There are two endpoints. When first launching that shell script perhaps wait a bit if you encounter a 500 error code.

## 1. /embed

This endpoint is used to first generate embeddings and insert them into the database to enable search functionality.

This is how to add a text:
'''
curl -X POST http://localhost:8081/embed \
 -H "Content-Type: application/json" \
 -d '{"text": "text to be embedded"}'
'''

This is how to add an image:

'''
curl -X POST http://localhost:8081/embed \
 -H "Content-Type: application/json" \
 -d '{"image": "base64 encoded image goes here"}'
'''

## 2. /search

This endpoint is used exclusively for searching the database for content with similar embedding, the top 10 results shall be returned (Can be text and images):

This is how to search based off of a text:
'''
curl -X POST http://localhost:8081/search \
 -H "Content-Type: application/json" \
 -d '{"text": "search string goes here"}'
'''

This is how to search based on an image:

'''
curl -X POST http://localhost:8081/search \
 -H "Content-Type: application/json" \
 -d '{"image": "base64 encoded image goes here"}'
'''
