class testeq:
    """Two instances of customized classes are judged equal   
    only if they have the same id.  
    """

    def __init__(self,x,y):
        self.xin=x
        self.yin=y


tx=testeq(1,2)
ty=testeq(1,2)
print(tx==ty)  # False
