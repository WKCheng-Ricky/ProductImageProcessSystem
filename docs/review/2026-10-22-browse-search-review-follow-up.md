# Review Process

There are 2 left behavior is strange and would like to fix here.

First is the "Not Implemented" error 

"""
product-image-processor-1 |ts=2026-10-23 10:18:44,244 level=INFO logger=src.pipeline.milvus_db msg='Using existing groups collection: product_groups'
product-image-processor-1 |ts=2026-10-23 10:18:44,253 level=ERROR logger=grpc._server msg='Exception calling application: Method not implemented!'
product-image-processor-1 |Traceback (most recent call last):
product-image-processor-1 |  File "/opt/venv/lib/python3.12/site-packages/grpc/_server.py", line 608, in _call_behavior
product-image-processor-1 |    response_or_iterator = behavior(argument, context)
product-image-processor-1 |                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
product-image-processor-1 |  File "/opt/venv/lib/python3.12/site-packages/pymilvus/grpc_gen/milvus_pb2_grpc.py", line 1357, in AllocTimestamp
product-image-processor-1 |    raise NotImplementedError('Method not implemented!')
product-image-processor-1 |NotImplementedError: Method not implemented!
"""

Altho it is out-of-scope, I think it should just be fixed in this branch to avoid creating another 'fix' branch just to fix this and write a bunch of document 

Second is browse-search weird behavior. So now, when there is an existing grouping, it will show the grouping result, right? However, after adding some photo, it still show the previous grouping result, which is misleading, as the grouping is stale (not full db group). I thought the test cover this ... but it doesn't work well. If new photo is added, the grouping no longer useful in browse-search and should use random again. 