from collections import namedtuple

Customer = namedtuple('Customer', 'name fidelity')


class LineItem:
    """each item one buys"""

    def __init__(self, product, quantity, price):
        self.product = product
        self.quantity = quantity
        self.price = price
    
    def total(self):
        return self.price * self.quantity


class Order:
    """an order"""

    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = list(cart)
        self.promotion = promotion
    
    def total(self):
        if not hasattr(self, '__total'):
            self.__total = sum(item.total() for item in self.cart)
        return self.__total
    
    def due(self):
        if self.promotion is None:
            discount = 0
        else:
            discount = self.promotion(self)
        return self.total() - discount
    
    def __repr__(self):
        """To define the text output when printing 'Order'.  """
        fmt = '<Order total: {:0.2f} due: {:0.2f}>'
        return fmt.format(self.total(), self.due())


promos=[]

def promotion(pro_func):
    """definition of the decorator"""
    promos.append(pro_func)
    return pro_func


@promotion
def fidelity(order):
    """.05 discount for 1000+ fidelity"""
    if order.customer.fidelity >= 1000:
        return order.total() * .05
    else:
        return 0

@promotion
def bulk_item(order):
    """.1 discount for ..."""
    discount = 0
    for item in order.cart:
        if item.quantity >= 20:
            discount += item.total() * .01
    return discount
    
def best_promo(order):
    """choose the best promotion avaialble"""
    return max(promo(order) for promo in promos)


def main():
    joe = Customer('Joe', 0)
    ann = Customer('Ann', 1150)

    cart = [LineItem('banana', 15, 3.5),
            LineItem('apple', 25, .5)]
    

    print(Order(joe, cart, best_promo))
    print(Order(ann, cart, bulk_item))
    print(Order(ann, cart, best_promo))

if __name__ == '__main__':
    main()
"""output:
<Order total: 65.00 due: 64.88>
<Order total: 65.00 due: 64.88>
<Order total: 65.00 due: 61.75>
"""


