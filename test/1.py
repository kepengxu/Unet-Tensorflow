class o1(object):
    def __init__(self,x):
        self.x=x
        pass
    def __call__(self, y,*args, **kwargs):
        return self.x*y*y

m=o1(1)(4)
print m