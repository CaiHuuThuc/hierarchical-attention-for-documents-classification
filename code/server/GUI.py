from tkinter import *

import grpc

# import the generated classes
import calculator_pb2
import calculator_pb2_grpc
import re
from tkinter.scrolledtext import ScrolledText



LIMIT_WIDTH = 80

def analysis():
    global root
    global e1
    current_row = 4
    sentence = e1.get("1.0",END).strip()
    
    req = calculator_pb2.Req(sentence=sentence)
    response = stub.analysis(req)

    text = eval(response.token)
    
    lengs = []
    for sent in text:
        lengs.append(len(sent))
    
    
    alphas_words = []
    alpha_words = []
    for idx, alphas in enumerate(eval(response.alpha_words)):
        for subidx, alpha in enumerate(alphas):
            if subidx < lengs[idx]:
                alpha_words.append(alpha)
            else:
                if alpha_words != []:
                    alphas_words.append(alpha_words)
                    alpha_words = []
                
    if alpha_words != []:
        alphas_words.append(alpha_words)

    # for alpha in alphas_words:
    #     print(alpha)
    print('\n\n')
    print("Predict: ", response.label)
    
    alphas_sentences = eval(response.alpha_sentences)[0]
    print("\n\n\n")
    strings = []

    for idx, sentence in enumerate(text):
        string = "{:<5.3f} ".format(alphas_sentences[idx])
        alpha_string = "{:<5} ".format("")
        for subidx, word in enumerate(sentence):
            alpha_string += "{:<12.3f}".format(alphas_words[idx][subidx])
            
            string += "{:<12}".format(word)
                
            
        strings.append(string + "\n" + alpha_string)
    for s in strings:
        print(s)
        print("\n")

    


if __name__ == '__main__':
    root = Tk()
    root.geometry("1000x800+100+100")
    
    root_frame = Frame(root, height=25, width=210)
    Label(root_frame, text='Enter text', height=3, pady=5).grid(row=0)
    root_frame.grid(row=0)

    e1 = Text(root_frame, width=80, height=10, pady=5)

    e1.grid(row=0, column=1)

    button = Button(root_frame, text="Analysis", command=analysis, pady=2)
    button.grid(row=1, column=1)

    
    # open a gRPC channel
    channel = grpc.insecure_channel('0.0.0.0:50051')

    # create a stub (client)
    stub = calculator_pb2_grpc.CalculatorStub(channel)
    
    

    root.mainloop()