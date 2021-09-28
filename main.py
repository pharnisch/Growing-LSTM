import torch


print("test")

lstm = torch.nn.LSTM(
    input_size=3,
    hidden_size=2,
    num_layers=1,
    bias=False,
    batch_first=True,  # does not apply to hidden or cell states
    dropout=0,
    bidirectional=False,
    proj_size=0,
)
print(list(lstm.named_parameters()))
lstm2 = torch.nn.LSTM(
    input_size=3,
    hidden_size=3,
    num_layers=1,
    bias=False,
    batch_first=True,  # does not apply to hidden or cell states
    dropout=0,
    bidirectional=False,
    proj_size=0,
)
print(list(lstm2.parameters()))

t = torch.tensor([[[1,2,3]]], dtype=torch.double)
input = torch.randn(1, 1, 3)
#print(t)
#print(input)

print(lstm(input))
print(lstm2(input))


param_list = list(lstm.parameters())
#print(list(lstm.named_parameters()))
lstm2.weight_ih_l0.data = torch.cat((param_list[0], torch.randn((4,3))), 0)

print(param_list[1])
rand = torch.randn((8, 1))
print(rand)
rand2 = torch.randn((4,3))
print(rand2)



new_hidden = torch.cat((param_list[1], rand), 1)
new_hidden = torch.cat((new_hidden, rand2), 0)

print(new_hidden)

lstm2.weight_hh_l0.data = new_hidden


print(list(lstm2.parameters()))


print(lstm2(input))