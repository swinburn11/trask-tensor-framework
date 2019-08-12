
from neuralflow.Tensor import Tensor


def test_tensor_add():
    result = Tensor([1, 2, 3]) + Tensor([10, 11, 12])
    assert result == Tensor([11, 13, 15])


def test_tensor_add_self():
    tensor = Tensor([1, 2, 3])
    assert Tensor([2, 4, 6]) == (tensor + tensor)


def test_gradient_propagation():
    a = Tensor([1, 2, 3])
    b = Tensor([10, 10, 10])
    c = a + b
    c.backward(Tensor([5, 6, 7]))
    assert a.grad == Tensor([5, 6, 7])
    assert c.creation_op == "add"


def test_correct_gradient_calculation():
    a = Tensor([1, 1, 1], autograd=True)
    b = Tensor([2, 2, 2], autograd=True)
    c = Tensor([2, 2, 2], autograd=True)
    d = a + b
    e = b + c
    f = d + e
    f.backward(Tensor([1, 1, 1]))
    assert b.grad == Tensor([2, 2, 2])


def test_neg_gradient_calculation():
    a = Tensor([1, 1, 1], autograd=True)
    b = Tensor([2, 2, 2], autograd=True)
    c = Tensor([2, 2, 2], autograd=True)
    d = a + (-b)
    e = (-b) + c
    f = d + e
    f.backward(Tensor([1, 1, 1]))
    assert b.grad == Tensor([-2, -2, -2])
