class o1(object):
    def __init__(self,x):
        self.x=x
        pass
    def __call__(self, y,*args, **kwargs):
        return self.x*y*y

m=o1(1)(4)
print m

def generator():
    for i in range(1000):
        yield i*i
a=generator()
for k in range(10):
    print next(a)