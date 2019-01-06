import grpc

# import the generated classes
import calculator_pb2
import calculator_pb2_grpc

# open a gRPC channel
channel = grpc.insecure_channel('localhost:50051')

# create a stub (client)
stub = calculator_pb2_grpc.CalculatorStub(channel)

req = calculator_pb2.Req(sentence="thrombocytopenia , or decreased platelet count , was reported in clinical trials of kadcyla ( 103 of 884 treated patients with > = grade 3 ; 283 of 884 treated patients with any grade ) . ")

response = stub.getNER(req)

print(response.token)
print(response.label)