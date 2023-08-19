class Strkey(dict):

    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
        
    def __contains__(self, key):
        return key in self.keys() or str(key) in self.keys()


if __name__=="__main__":
    d=Strkey([('2', 'two'),('3', 'three')])  # dict_keys(['2', '3'])
    print(d.keys())
    print(d[2])  # two
    print(d[3])  # three
    # print(d[4]) -> KeyError: '4'

    print(d.get('2'))  # two
    print(d.get(3))  # three
    # print(d.get(4)) -> None
    
    print(2 in d)  # True
    # print(4 in d) -> False
