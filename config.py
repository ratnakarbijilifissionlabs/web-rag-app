from opensearchpy import OpenSearch
import boto3
 
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
)

host = 'localhost'
port = 9200
auth = ('admin', 'FissionLabs@123') 

client = OpenSearch(
    hosts = [{'host': host, 'port': port}],
    http_compress = True,
    http_auth = auth,
    use_ssl = True,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False
)

index_name = "scraped-data-test"